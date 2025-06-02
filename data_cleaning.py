import re
import json

class ProductionDataCleaner:
    def __init__(self):
        self.company_indicators = {
            'inc', 'incorporated', 'llc', 'corp', 'corporation', 'ltd', 'limited',
            'company', 'co', 'enterprises', 'group', 'holdings', 'industries',
            'international', 'systems', 'solutions', 'services', 'technologies',
            '&', 'and', 'associates', 'partners', 'consulting'
        }
        self.person_prefixes = {
            'mr', 'mrs', 'ms', 'dr', 'prof', 'professor', 'sir', 'madam',
            'miss', 'master', 'captain', 'colonel', 'major', 'lieutenant'
        }

    def is_likely_company(self, name_text):
        name_lower = name_text.lower()
        for indicator in self.company_indicators:
            if indicator in name_lower:
                return True
        for prefix in self.person_prefixes:
            if name_lower.startswith(prefix + ' '):
                return False
        if name_text.isupper() and len(name_text.split()) > 1:
            return True
        if any(char.isdigit() for char in name_text):
            return True
        return False

    def is_valid_email(self, email):
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

    def is_valid_phone(self, phone):
        digits = re.sub(r'[^\d]', '', phone)
        return 7 <= len(digits) <= 15

    def clean_census_data(self, census_data):
        cleaned_examples = []
        stats = {'total': 0, 'kept': 0, 'discarded_name_pollution': 0, 'discarded_invalid_contact': 0}
        for example in census_data:
            stats['total'] += 1
            try:
                user_text = example['user']
                assistant_data = json.loads(example['assistant'])
                name = assistant_data.get('name', '').strip()
                email = assistant_data.get('email', '').strip()
                phone = assistant_data.get('phone_number', '').strip()
                if not email and not phone:
                    stats['discarded_invalid_contact'] += 1
                    continue
                email_valid = self.is_valid_email(email) if email and email != 'nan' else False
                phone_valid = self.is_valid_phone(phone) if phone and phone != 'nan' else False
                if not email_valid and not phone_valid:
                    stats['discarded_invalid_contact'] += 1
                    continue
                if name and name != 'nan':
                    if self.is_likely_company(name):
                        stats['discarded_name_pollution'] += 1
                        continue
                tokens = user_text.replace(',', ' , ').replace('.', ' . ').replace('(', ' ( ').replace(')', ' ) ').split()
                tags = [0] * len(tokens)
                if email_valid:
                    # Find all email-related tokens
                    email_groups = []
                    i = 0
                    
                    while i < len(tokens):
                        token = tokens[i].lower()
                        
                        # Look for email start (contains @ or starts an email sequence)
                        if '@' in token or (i < len(tokens) - 2 and 
                                          tokens[i+1] == '@' and 
                                          tokens[i+2].lower().replace('.', '') in ['com', 'net', 'org', 'gov', 'edu']):
                            email_group = []
                            
                            # Collect email parts (may be separated by spaces)
                            while i < len(tokens):
                                current_token = tokens[i]
                                
                                # Check if token is email-related
                                is_email_part = ('@' in current_token or 
                                               current_token == '.' or
                                               current_token.lower() in ['com', 'net', 'org', 'gov', 'edu', 'hotmail', 'gmail', 'yahoo'] or
                                               (current_token.isalnum() and len(current_token) > 1))
                                
                                if is_email_part:
                                    email_group.append(i)
                                    i += 1
                                    # Continue if next token might be part of email
                                    if (i < len(tokens) and 
                                        (tokens[i] == '.' or 
                                         tokens[i].lower() in ['com', 'net', 'org', 'gov', 'edu'] or
                                         '@' in tokens[i])):
                                        continue
                                    else:
                                        break
                                else:
                                    break
                            
                            # Add valid email groups (must contain @)
                            if email_group and any('@' in tokens[idx] for idx in email_group):
                                email_groups.append(email_group)
                        else:
                            i += 1
                    
                    # Apply BIO tagging to each email group independently
                    for group in email_groups:
                        if group:
                            tags[group[0]] = 9   # B-EMAIL
                            for idx in group[1:]:
                                tags[idx] = 10   # I-EMAIL

                if phone_valid:
                    phone_digits = re.sub(r'[^\d]', '', phone)
                    
                    # Find all potential phone number groups in the text
                    phone_groups = []
                    i = 0
                    
                    while i < len(tokens):
                        token = tokens[i]
                        token_digits = re.sub(r'[^\d]', '', token)
                        
                        # Look for start of phone number (opening paren or digit sequence)
                        if token == '(' or (len(token_digits) >= 3 and token_digits in phone_digits):
                            phone_group = []
                            
                            # Collect consecutive phone-related tokens
                            while i < len(tokens):
                                current_token = tokens[i]
                                current_digits = re.sub(r'[^\d]', '', current_token)
                                
                                # Check if token is phone-related
                                is_phone_digit = len(current_digits) >= 3 and current_digits in phone_digits
                                is_phone_punct = current_token in ['(', ')', '-', '.', '+']
                                
                                if is_phone_digit or is_phone_punct:
                                    # But avoid decimal numbers (check context)
                                    if current_token == '.':
                                        # Only include if between phone digits
                                        prev_is_phone = (i > 0 and i-1 < len(tokens) and 
                                                        len(re.sub(r'[^\d]', '', tokens[i-1])) >= 3)
                                        next_is_phone = (i < len(tokens)-1 and 
                                                        len(re.sub(r'[^\d]', '', tokens[i+1])) >= 3)
                                        if prev_is_phone and next_is_phone:
                                            phone_group.append(i)
                                    else:
                                        phone_group.append(i)
                                    i += 1
                                else:
                                    break
                            
                            # Add valid phone groups (must have at least one digit token)
                            if phone_group and any(len(re.sub(r'[^\d]', '', tokens[idx])) >= 3 
                                                 for idx in phone_group):
                                # Validate total phone number has at least 7 digits
                                total_digits = ''.join(re.sub(r'[^\d]', '', tokens[idx]) 
                                                     for idx in phone_group)
                                if len(total_digits) >= 7:
                                    phone_groups.append(phone_group)
                        else:
                            i += 1
                    
                    # Apply BIO tagging to each phone group independently
                    for group in phone_groups:
                        if group:
                            tags[group[0]] = 11  # B-PHONE
                            for idx in group[1:]:
                                tags[idx] = 12  # I-PHONE
                if any(tag > 0 for tag in tags):
                    cleaned_examples.append({'tokens': tokens, 'ner_tags': tags})
                    stats['kept'] += 1
                else:
                    stats['discarded_invalid_contact'] += 1
            except Exception as e:
                stats['discarded_invalid_contact'] += 1
                continue
        print(f"📊 Census Data Cleaning Results:")
        print(f"   Total examples: {stats['total']:,}")
        print(f"   Kept (clean): {stats['kept']:,}")
        print(f"   Discarded (name pollution): {stats['discarded_name_pollution']:,}")
        print(f"   Data quality: {stats['kept']/stats['total']*100:.1f}%")
        return cleaned_examples
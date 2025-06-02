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
                    email_lower = email.lower()
                    for i, token in enumerate(tokens):
                        if '@' in token.lower() and any(part in token.lower() for part in email_lower.split('@')):
                            tags[i] = 9
                if phone_valid:
                    phone_digits = re.sub(r'[^\d]', '', phone)
                    for i, token in enumerate(tokens):
                        token_digits = re.sub(r'[^\d]', '', token)
                        if len(token_digits) >= 3 and token_digits in phone_digits:
                            tags[i] = 11
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
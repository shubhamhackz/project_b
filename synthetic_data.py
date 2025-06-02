import random
import re

class AdvancedSyntheticGenerator:
    def __init__(self):
        self.first_names = [
            "John", "Jane", "Michael", "Sarah", "David", "Emily", "Robert", "Ashley",
            "William", "Jessica", "James", "Amanda", "Daniel", "Jennifer", "Christopher", "Lisa",
            "Ahmed", "Fatima", "Liu", "Chen", "Hiroshi", "Akiko", "Giovanni", "Maria",
            "José", "Carmen", "Pierre", "Marie", "Olaf", "Ingrid", "Raj", "Priya"
        ]
        self.last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
            "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
            "Wang", "Zhang", "Kumar", "Patel", "Singh", "Mueller", "Schmidt", "Rossi", "Ferrari"
        ]
        self.company_names = [
            "Tech Solutions", "Data Systems", "Innovation Corp", "Global Dynamics", "Future Labs",
            "Smart Technologies", "Digital Partners", "Advanced Analytics", "Cloud Networks",
            "AI Innovations", "Quantum Computing", "Cyber Security", "Mobile Apps", "Web Services"
        ]
        self.company_suffixes = ["Inc", "LLC", "Corp", "Ltd", "Co", "Group", "Partners", "Associates"]
        self.email_domains = [
            "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "company.com",
            "business.org", "tech.net", "solutions.co", "global.biz", "enterprise.io"
        ]
        self.templates = [
            "{name} works at {company}. Contact via {email} or {phone}.",
            "Reach {name} from {company} at {email} or call {phone}.",
            "{company} representative {name}: {email}, phone: {phone}.",
            "Contact info for {name} ({company}): Email {email}, Phone {phone}.",
            "{name} - {company} - {email} - {phone}",
            "For assistance, contact {name} at {company}. Email: {email}, Phone: {phone}."
        ]

    def generate_realistic_examples(self, count=15000):
        examples = []
        for _ in range(count):
            first_name = random.choice(self.first_names)
            last_name = random.choice(self.last_names)
            name = f"{first_name} {last_name}"
            company_base = random.choice(self.company_names)
            company_suffix = random.choice(self.company_suffixes)
            company = f"{company_base} {company_suffix}"
            email_user = f"{first_name.lower()}.{last_name.lower()}"
            if random.random() < 0.3:
                email_user = f"{first_name[0].lower()}{last_name.lower()}"
            domain = random.choice(self.email_domains)
            email = f"{email_user}@{domain}"
            area_code = random.randint(200, 999)
            prefix = random.randint(200, 999)
            number = random.randint(1000, 9999)
            phone_formats = [
                f"({area_code}) {prefix}-{number}",
                f"{area_code}-{prefix}-{number}",
                f"{area_code}.{prefix}.{number}",
                f"+1 {area_code} {prefix} {number}"
            ]
            phone = random.choice(phone_formats)
            template = random.choice(self.templates)
            text = template.format(name=name, company=company, email=email, phone=phone)
            tokens = text.replace(',', ' , ').replace('.', ' . ').replace('(', ' ( ').replace(')', ' ) ').replace(':', ' : ').split()
            tags = [0] * len(tokens)
            
            # Track entity positions for proper BIO tagging
            person_tokens = []
            org_tokens = []
            email_tokens = []
            phone_tokens = []
            
            # Find all entity tokens first - CONTEXT-AWARE VERSION
            for i, token in enumerate(tokens):
                token_lower = token.lower()
                
                # Email detection FIRST (highest priority but context-aware)
                if '@' in token:
                    email_tokens.append(i)
                
            # Find tokens near @ symbol for email context
            email_context_indices = set()
            for email_idx in email_tokens:
                # Look 3 tokens before and after @ symbol for email components
                for j in range(max(0, email_idx - 3), min(len(tokens), email_idx + 4)):
                    email_context_indices.add(j)
            
            # Second pass: context-aware detection
            for i, token in enumerate(tokens):
                token_lower = token.lower()
                
                # Email detection (already found @ symbols)
                if '@' in token:
                    continue  # Already added
                
                # Email components (only near @ symbol)
                elif i in email_context_indices:
                    if any(email_part in token_lower for email_part in email.lower().replace('@', '.').split('.')):
                        email_tokens.append(i)
                    elif token in ['.'] and i > 0 and i < len(tokens) - 1:
                        # Punctuation between email parts
                        prev_in_email = (i-1) in email_context_indices
                        next_in_email = (i+1) in email_context_indices
                        if prev_in_email and next_in_email:
                            email_tokens.append(i)
                
                # Phone detection (digits with 3+ characters)
                elif len(re.sub(r'[^\d]', '', token)) >= 3:
                    phone_tokens.append(i)
                
                # Phone punctuation between digits
                elif token in ['(', ')', '-', '.', '+'] and i > 0 and i < len(tokens) - 1:
                    prev_has_digits = bool(re.search(r'\d', tokens[i-1]))
                    next_has_digits = bool(re.search(r'\d', tokens[i+1]))
                    if prev_has_digits and next_has_digits:
                        phone_tokens.append(i)
                
                # Phone trailing punctuation (like final '.' after phone)
                elif token in ['.', ')'] and i > 0:
                    prev_token_digits = re.sub(r'[^\d]', '', tokens[i-1])
                    if len(prev_token_digits) >= 3:  # Previous token was likely phone digits
                        phone_tokens.append(i)
                
                # Person name detection (ONLY if NOT in email context)
                elif i not in email_context_indices:
                    if first_name.lower() in token_lower or last_name.lower() in token_lower:
                        # Additional check: prioritize capitalized formal names
                        if token[0].isupper():  # Capitalized = formal person name
                            person_tokens.append(i)
                
                # Organization detection (ONLY if NOT in email context)
                elif i not in email_context_indices:
                    if any(part.lower() in token_lower for part in company.lower().split()):
                        org_tokens.append(i)
            
            # Apply proper BIO tagging
            # Person entities (B-PER = 1, I-PER = 2)
            if person_tokens:
                person_tokens.sort()
                tags[person_tokens[0]] = 1  # B-PER
                for idx in person_tokens[1:]:
                    tags[idx] = 2  # I-PER
            
            # Organization entities (B-ORG = 3, I-ORG = 4)  
            if org_tokens:
                org_tokens.sort()
                tags[org_tokens[0]] = 3  # B-ORG
                for idx in org_tokens[1:]:
                    tags[idx] = 4  # I-ORG
            
            # Email entities (B-EMAIL = 9, I-EMAIL = 10)
            if email_tokens:
                email_tokens.sort()
                tags[email_tokens[0]] = 9  # B-EMAIL
                for idx in email_tokens[1:]:
                    tags[idx] = 10  # I-EMAIL
            
            # Phone entities (B-PHONE = 11, I-PHONE = 12)
            if phone_tokens:
                phone_tokens.sort()
                tags[phone_tokens[0]] = 11  # B-PHONE
                for idx in phone_tokens[1:]:
                    tags[idx] = 12  # I-PHONE
            
            examples.append({'tokens': tokens, 'ner_tags': tags})
        return examples
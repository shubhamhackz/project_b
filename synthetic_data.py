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
            for i, token in enumerate(tokens):
                token_lower = token.lower()
                if first_name.lower() in token_lower:
                    if i > 0 and tags[i-1] in [1, 2]:
                        tags[i] = 2
                    else:
                        tags[i] = 1
                elif last_name.lower() in token_lower:
                    if i > 0 and tags[i-1] in [1, 2]:
                        tags[i] = 2
                    else:
                        tags[i] = 1
                elif any(part.lower() in token_lower for part in company.lower().split()):
                    if i > 0 and tags[i-1] in [3, 4]:
                        tags[i] = 4
                    else:
                        tags[i] = 3
                elif '@' in token:
                    tags[i] = 9
                elif len(re.sub(r'[^\d]', '', token)) >= 3:
                    if i > 0 and tags[i-1] in [11, 12]:
                        tags[i] = 12
                    else:
                        tags[i] = 11
            examples.append({'tokens': tokens, 'ner_tags': tags})
        return examples
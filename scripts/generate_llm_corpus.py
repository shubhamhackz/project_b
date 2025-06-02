#!/usr/bin/env python3
"""
Generate high-quality NER training corpus using LLM-style structured generation
This simulates what you'd get from GPT-4/Claude for NER training data
"""

import json
import random
import re
from typing import List, Dict, Tuple

class LLMCorpusGenerator:
    def __init__(self):
        # Realistic entity pools
        self.persons = [
            "Dr. Sarah Chen", "Michael Rodriguez", "Prof. Emma Thompson", "David Kim", 
            "Jennifer Martinez", "Dr. Ahmed Hassan", "Maria Gonzalez", "Robert Johnson",
            "Lisa Wang", "James Anderson", "Dr. Priya Patel", "Carlos Rivera",
            "Anna Kowalski", "Mohammed Al-Rashid", "Elena Petrov", "Hiroshi Tanaka"
        ]
        
        self.organizations = [
            "Microsoft Corporation", "Stanford University", "Goldman Sachs", "NASA",
            "World Health Organization", "Google DeepMind", "MIT Technology Review",
            "United Nations", "Tesla Inc", "Harvard Medical School", "OpenAI",
            "European Central Bank", "Netflix Studios", "Amazon Web Services"
        ]
        
        self.locations = [
            "San Francisco", "New York City", "London", "Tokyo", "Berlin",
            "Singapore", "Sydney", "Toronto", "Paris", "Seoul", "Mumbai",
            "São Paulo", "Dubai", "Stockholm", "Zurich", "Tel Aviv"
        ]
        
        self.misc_entities = [
            "iPhone 15", "Tesla Model S", "COVID-19", "Python 3.11", "ChatGPT",
            "Nobel Prize", "Olympics 2024", "NASA Artemis", "SpaceX Falcon 9",
            "MacBook Pro", "Azure OpenAI", "GPT-4", "BERT model", "transformer architecture"
        ]
        
        # Email and phone generation
        self.email_domains = [
            "gmail.com", "outlook.com", "company.com", "university.edu", "research.org",
            "tech.ai", "startup.io", "global.net", "enterprise.biz", "institute.edu"
        ]
        
        # Sophisticated text templates with varied complexity
        self.templates = [
            # Business contexts
            "Please contact {person} from {organization} regarding the {misc} project. Email: {email}, Phone: {phone}.",
            "{person} serves as CEO of {organization} based in {location}. Reach out via {email} or {phone}.",
            "The research team led by {person} at {organization} published findings on {misc}. Contact: {email}, {phone}.",
            
            # Academic contexts  
            "Prof. {person} from {organization} in {location} will present on {misc}. Details: {email}, Phone: {phone}.",
            "{person} completed their PhD at {organization} studying {misc}. Current email: {email}, Tel: {phone}.",
            
            # News/Media contexts
            "Breaking: {person} of {organization} announces {misc} breakthrough in {location}. Press contact: {email}, {phone}.",
            "{organization} spokesperson {person} confirmed the {misc} launch. Media inquiries: {email} or {phone}.",
            
            # Conference/Event contexts
            "Keynote speaker {person} ({organization}) will discuss {misc} at the {location} summit. RSVP: {email}, Info: {phone}.",
            "Register for {person}'s workshop on {misc} hosted by {organization}. Contact: {email}, Phone: {phone}.",
            
            # Technical contexts
            "{person} from {organization} developed the {misc} framework. Technical support: {email}, Phone: {phone}.",
            "The {misc} system deployed by {organization} in {location} was designed by {person}. Contact: {email}, {phone}.",
            
            # Simple contact formats
            "{person} - {organization} - {email} - {phone}",
            "Contact: {person}, {organization}, {location}. Email: {email}, Tel: {phone}.",
            
            # Complex multi-entity scenarios
            "{person} and colleagues from {organization} collaborated with teams in {location} on {misc}. Lead contact: {email}, Office: {phone}.",
            "Joint venture between {person} ({organization}) and {location}-based partners focuses on {misc}. Email: {email}, Phone: {phone}."
        ]

    def generate_realistic_email(self, person_name: str) -> str:
        """Generate realistic email based on person name"""
        name_parts = person_name.replace("Dr. ", "").replace("Prof. ", "").split()
        first = name_parts[0].lower()
        last = name_parts[-1].lower()
        
        formats = [
            f"{first}.{last}",
            f"{first[0]}{last}",
            f"{first}{last[0]}",
            f"{first}_{last}",
            f"{last}.{first}"
        ]
        
        user = random.choice(formats)
        domain = random.choice(self.email_domains)
        return f"{user}@{domain}"

    def generate_realistic_phone(self) -> str:
        """Generate realistic phone numbers in various formats"""
        area = random.randint(200, 999)
        prefix = random.randint(200, 999) 
        number = random.randint(1000, 9999)
        
        formats = [
            f"({area}) {prefix}-{number}",
            f"+1 {area} {prefix} {number}",
            f"{area}-{prefix}-{number}",
            f"{area}.{prefix}.{number}",
            f"+1-{area}-{prefix}-{number}"
        ]
        
        return random.choice(formats)

    def tokenize_and_tag(self, text: str, entities: Dict[str, str]) -> Tuple[List[str], List[int]]:
        """Tokenize text and apply NER tags with proper BIO tagging"""
        # Normalize text for tokenization
        normalized = text
        for punct in [',', '.', '(', ')', ':', '-', '+']:
            normalized = normalized.replace(punct, f' {punct} ')
        
        tokens = normalized.split()
        tags = [0] * len(tokens)  # O tag
        
        # Tag entities with proper BIO sequence
        for entity_type, entity_value in entities.items():
            if not entity_value:
                continue
                
            # Handle email entities
            if entity_type == 'email':
                self._tag_email_entity(tokens, tags, entity_value)
            
            # Handle phone entities  
            elif entity_type == 'phone':
                self._tag_phone_entity(tokens, tags, entity_value)
                
            # Handle named entities (PERSON, ORG, LOC, MISC)
            else:
                self._tag_named_entity(tokens, tags, entity_value, entity_type)
        
        return tokens, tags

    def _tag_email_entity(self, tokens: List[str], tags: List[int], email: str):
        """Tag email entities with B-EMAIL (9) and I-EMAIL (10)"""
        email_parts = email.replace('@', ' @ ').replace('.', ' . ').split()
        
        # Find email sequence in tokens
        for i in range(len(tokens) - len(email_parts) + 1):
            if self._matches_sequence(tokens[i:i+len(email_parts)], email_parts):
                tags[i] = 9  # B-EMAIL
                for j in range(1, len(email_parts)):
                    tags[i + j] = 10  # I-EMAIL
                break

    def _tag_phone_entity(self, tokens: List[str], tags: List[int], phone: str):
        """Tag phone entities with B-PHONE (11) and I-PHONE (12)"""
        # Normalize phone for matching
        phone_normalized = phone
        for punct in ['(', ')', '-', '.', '+']:
            phone_normalized = phone_normalized.replace(punct, f' {punct} ')
        phone_parts = phone_normalized.split()
        
        # Find phone sequence in tokens
        for i in range(len(tokens) - len(phone_parts) + 1):
            if self._matches_sequence(tokens[i:i+len(phone_parts)], phone_parts):
                tags[i] = 11  # B-PHONE
                for j in range(1, len(phone_parts)):
                    tags[i + j] = 12  # I-PHONE
                break

    def _tag_named_entity(self, tokens: List[str], tags: List[int], entity: str, entity_type: str):
        """Tag named entities (PERSON, ORG, LOC, MISC)"""
        entity_clean = entity.replace("Dr. ", "").replace("Prof. ", "")
        entity_words = entity_clean.split()
        
        tag_map = {
            'person': (1, 2),    # B-PER, I-PER
            'organization': (3, 4),  # B-ORG, I-ORG
            'location': (5, 6),      # B-LOC, I-LOC
            'misc': (7, 8)           # B-MISC, I-MISC
        }
        
        b_tag, i_tag = tag_map.get(entity_type, (0, 0))
        if b_tag == 0:
            return
            
        # Find entity sequence in tokens
        for i in range(len(tokens) - len(entity_words) + 1):
            if self._matches_sequence(tokens[i:i+len(entity_words)], entity_words, fuzzy=True):
                tags[i] = b_tag
                for j in range(1, len(entity_words)):
                    tags[i + j] = i_tag
                break

    def _matches_sequence(self, token_seq: List[str], target_seq: List[str], fuzzy: bool = False) -> bool:
        """Check if token sequence matches target sequence"""
        if len(token_seq) != len(target_seq):
            return False
            
        for t1, t2 in zip(token_seq, target_seq):
            if fuzzy:
                if t1.lower().strip('.,()') != t2.lower().strip('.,()'):
                    return False
            else:
                if t1.strip() != t2.strip():
                    return False
        return True

    def generate_corpus(self, num_examples: int = 5000) -> List[Dict]:
        """Generate high-quality LLM-style NER corpus"""
        examples = []
        
        for i in range(num_examples):
            # Select entities for this example
            person = random.choice(self.persons)
            org = random.choice(self.organizations) 
            location = random.choice(self.locations)
            misc = random.choice(self.misc_entities)
            
            # Generate contact info
            email = self.generate_realistic_email(person)
            phone = self.generate_realistic_phone()
            
            # Create text from template
            template = random.choice(self.templates)
            text = template.format(
                person=person,
                organization=org, 
                location=location,
                misc=misc,
                email=email,
                phone=phone
            )
            
            # Tokenize and tag
            entities = {
                'person': person,
                'organization': org,
                'location': location, 
                'misc': misc,
                'email': email,
                'phone': phone
            }
            
            tokens, tags = self.tokenize_and_tag(text, entities)
            
            examples.append({
                'tokens': tokens,
                'ner_tags': tags,
                'source': 'llm_generated',
                'template_id': self.templates.index(template)
            })
            
            # Progress indicator
            if (i + 1) % 1000 == 0:
                print(f"Generated {i + 1}/{num_examples} examples...")
        
        return examples

def main():
    print("🤖 LLM-Style NER Corpus Generation")
    print("=" * 50)
    
    generator = LLMCorpusGenerator()
    
    # Generate corpus
    print("🔄 Generating high-quality NER training data...")
    corpus = generator.generate_corpus(num_examples=5000)
    
    # Save to file
    output_file = "data/llm_generated.json"
    with open(output_file, 'w') as f:
        json.dump(corpus, f, indent=2)
    
    print(f"✅ Generated {len(corpus)} examples")
    print(f"💾 Saved to: {output_file}")
    
    # Show sample
    sample = random.choice(corpus)
    print(f"\n📝 Sample:")
    print(f"   Tokens: {' '.join(sample['tokens'][:15])}...")
    print(f"   Tags: {sample['ner_tags'][:15]}...")
    
    # Show label distribution
    label_counts = {}
    for example in corpus:
        for tag in example['ner_tags']:
            label_counts[tag] = label_counts.get(tag, 0) + 1
    
    label_names = {
        0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG',
        5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC', 
        9: 'B-EMAIL', 10: 'I-EMAIL', 11: 'B-PHONE', 12: 'I-PHONE'
    }
    
    print(f"\n📊 Label Distribution:")
    for tag_id, count in sorted(label_counts.items()):
        label = label_names.get(tag_id, f'UNK-{tag_id}')
        print(f"   {label}: {count:,}")

if __name__ == "__main__":
    main() 
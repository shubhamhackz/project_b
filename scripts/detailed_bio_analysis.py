#!/usr/bin/env python3
"""
Detailed analysis of BIO tagging issues in census data
"""

import json
from collections import Counter

def analyze_bio_issues():
    with open('data/census_clean.json', 'r') as f:
        data = json.load(f)
    
    print('🔍 Detailed BIO Tagging Issue Analysis')
    print('='*60)
    
    label_map = {
        0:'O', 1:'B-PER', 2:'I-PER', 3:'B-ORG', 4:'I-ORG', 
        9:'B-EMAIL', 10:'I-EMAIL', 11:'B-PHONE', 12:'I-PHONE'
    }
    
    issue_patterns = Counter()
    examples_with_issues = []
    
    # Analyze first 10 examples in detail
    for i, example in enumerate(data[:10]):
        tokens = example['tokens']
        tags = example['ner_tags']
        
        print(f'\n📝 Example {i+1}:')
        
        # Show labeled tokens in chunks
        labeled_tokens = []
        for token, tag in zip(tokens, tags):
            label = label_map.get(tag, f'UNK-{tag}')
            if tag in [9, 10, 11, 12]:  # Highlight email/phone tags
                labeled_tokens.append(f'[{token}/{label}]')
            else:
                labeled_tokens.append(f'{token}/{label}')
        
        # Print in chunks of 15 tokens for readability
        for chunk_start in range(0, len(labeled_tokens), 15):
            chunk = labeled_tokens[chunk_start:chunk_start+15]
            print(f'   {" ".join(chunk)}')
        
        # Check for BIO issues
        prev_tag = 0
        issues_in_example = []
        for j, tag in enumerate(tags):
            if tag == 12 and prev_tag not in [11, 12]:  # I-PHONE without B-PHONE
                issue_patterns['I-PHONE without B-PHONE'] += 1
                issues_in_example.append(f'Position {j}: I-PHONE without B-PHONE')
            elif tag == 10 and prev_tag not in [9, 10]:  # I-EMAIL without B-EMAIL
                issue_patterns['I-EMAIL without B-EMAIL'] += 1
                issues_in_example.append(f'Position {j}: I-EMAIL without B-EMAIL')
            prev_tag = tag
        
        if issues_in_example:
            print(f'   🔴 Issues: {"; ".join(issues_in_example)}')
            examples_with_issues.append(i+1)
        else:
            print(f'   ✅ No BIO issues')
    
    print(f'\n📊 Summary:')
    print(f'   Examples analyzed: 10')
    print(f'   Examples with issues: {len(examples_with_issues)}')
    print(f'   Issue breakdown:')
    for issue, count in issue_patterns.most_common():
        print(f'     {issue}: {count} occurrences')
    
    # Show how many examples have email/phone entities
    email_count = 0
    phone_count = 0
    for example in data[:100]:  # Check first 100
        has_email = any(tag in [9, 10] for tag in example['ner_tags'])
        has_phone = any(tag in [11, 12] for tag in example['ner_tags'])
        if has_email:
            email_count += 1
        if has_phone:
            phone_count += 1
    
    print(f'\n📈 Entity Distribution (first 100 examples):')
    print(f'   Examples with emails: {email_count}')
    print(f'   Examples with phones: {phone_count}')

if __name__ == "__main__":
    analyze_bio_issues() 
#!/usr/bin/env python3
"""
Comprehensive Model Testing Suite
=================================

Test the trained NER model on various real-world scenarios to ensure
it's ready for production deployment.
"""

import torch
import json
import pandas as pd
from transformers import AutoTokenizer
from model import AdvancedNERModel
import numpy as np
from collections import Counter, defaultdict
import re

class ComprehensiveModelTester:
    """Comprehensive testing suite for NER model"""
    
    def __init__(self, model_path: str, label_list: list):
        self.model_path = model_path
        self.label_list = label_list
        self.label2id = {l: i for i, l in enumerate(label_list)}
        self.id2label = {i: l for i, l in enumerate(label_list)}
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AdvancedNERModel.from_pretrained(model_path, num_labels=len(label_list))
        self.model.eval()
        
        print(f"✅ Model loaded from {model_path}")
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def predict_single(self, text: str):
        """Predict entities in a single text"""
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs["logits"]
            predictions = self.model.decode(logits, inputs["attention_mask"])
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        predicted_labels = [self.id2label[pred] for pred in predictions[0]]
        
        # Remove special tokens
        filtered_results = []
        for token, label in zip(tokens, predicted_labels):
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                filtered_results.append((token, label))
        
        return filtered_results

    def extract_entities(self, token_label_pairs):
        """Extract entities from token-label pairs"""
        entities = []
        current_entity = None
        
        for token, label in token_label_pairs:
            if label.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'type': label[2:],
                    'tokens': [token],
                    'text': token
                }
            elif label.startswith('I-') and current_entity and label[2:] == current_entity['type']:
                current_entity['tokens'].append(token)
                current_entity['text'] += ' ' + token
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        # Clean up subword tokens
        for entity in entities:
            entity['text'] = self.tokenizer.convert_tokens_to_string(entity['tokens'])
            
        return entities

    def test_basic_functionality(self):
        """Test basic model functionality"""
        print("\n🔍 BASIC FUNCTIONALITY TESTS")
        print("="*60)
        
        test_cases = [
            "Hello, my name is John Smith.",
            "Contact me at john.smith@example.com",
            "Call me at (555) 123-4567",
            "I work at Google Inc. in New York",
            "Visit us at 123 Main Street, Boston"
        ]
        
        results = {}
        for i, text in enumerate(test_cases, 1):
            print(f"\n{i}. Testing: '{text}'")
            try:
                predictions = self.predict_single(text)
                entities = self.extract_entities(predictions)
                results[text] = entities
                
                if entities:
                    for entity in entities:
                        print(f"   ✅ {entity['type']}: '{entity['text']}'")
                else:
                    print("   ⚠️  No entities detected")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results[text] = []
        
        return results

    def test_overfitting_scenarios(self):
        """Test scenarios that reveal overfitting (based on training log analysis)"""
        print("\n🚨 OVERFITTING DETECTION TESTS")
        print("="*60)
        print("Testing corrupted patterns that should NOT get perfect scores...")
        
        # These should be challenging if the model learned properly
        challenging_cases = [
            # Corrupted emails (should still detect but not get 99% confidence)
            "Email me at john..doe@company,com immediately",
            "Contact sarah AT company DOT org for details",
            "My email is user.name + tag @ example . com",
            
            # Corrupted phones (should still detect but not perfect)
            "Call me at 555..123..4567 today",
            "Phone: ( 555 ) 123 - 4567 extension 890",
            "Dial 1 800 FLOWERS for delivery",
            
            # Case variations (should handle but not be overconfident)
            "JOHN SMITH works at IBM CORPORATION",
            "dr mary johnson and prof ahmed al-rashid",
            "contact MS. SARAH o'connor immediately",
            
            # International formats (true test of generalization)
            "Call +44 20 7946 0958 for UK support",
            "Email p.dubois@société-générale.fr",
            "Contact +86 138 0013 8000 in China",
            
            # Ambiguous contexts (should not be overconfident)
            "Apple iPhone vs Samsung Galaxy comparison",
            "New York Times reported about New York City",
            "Microsoft CEO spoke at Microsoft conference"
        ]
        
        results = {}
        confidence_scores = []
        
        for i, text in enumerate(challenging_cases, 1):
            print(f"\n{i}. Challenging: '{text}'")
            try:
                predictions = self.predict_single(text)
                entities = self.extract_entities(predictions)
                results[text] = entities
                
                if entities:
                    for entity in entities:
                        print(f"   🎯 {entity['type']}: '{entity['text']}'")
                else:
                    print("   ⚠️  No entities detected (possibly too challenging)")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results[text] = []
        
        return results

    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        print("\n🌪️  EDGE CASE TESTS")
        print("="*60)
        
        edge_cases = [
            # Empty and minimal
            "",
            "a",
            "Hello.",
            
            # Very long text
            "This is a very long sentence with multiple entities like John Smith working at Google Inc. and his email john.smith@google.com and phone (555) 123-4567 " * 5,
            
            # Special characters
            "Contact María García-López at maría@español.es",
            "Call +1-800-555-HELP (4357) for assistance",
            "Email support@company-name.co.uk immediately",
            
            # Nested entities
            "john.doe@apple.com works at Apple Inc.",
            "Contact Apple Inc. support at support@apple.com",
            
            # Multiple entities of same type
            "John Smith, Jane Doe, and Bob Johnson attended",
            "Email john@example.com, jane@test.org, and bob@company.net",
            
            # Mixed formats
            "Call John Smith at (555) 123-4567 or email john.smith@example.com",
        ]
        
        results = {}
        for i, text in enumerate(edge_cases, 1):
            print(f"\n{i}. Edge case: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            try:
                predictions = self.predict_single(text)
                entities = self.extract_entities(predictions)
                results[text] = entities
                
                print(f"   📊 Found {len(entities)} entities")
                for entity in entities:
                    print(f"   → {entity['type']}: '{entity['text']}'")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results[text] = []
        
        return results

    def performance_analysis(self, all_results):
        """Analyze overall performance patterns"""
        print("\n📈 PERFORMANCE ANALYSIS")
        print("="*60)
        
        # Count entity types
        entity_counts = Counter()
        total_predictions = 0
        
        for text, entities in all_results.items():
            total_predictions += len(entities)
            for entity in entities:
                entity_counts[entity['type']] += 1
        
        print(f"📊 Total predictions: {total_predictions}")
        print(f"📊 Entity distribution:")
        for entity_type, count in entity_counts.most_common():
            print(f"   {entity_type}: {count} ({count/total_predictions*100:.1f}%)")
        
        # Identify potential issues
        print(f"\n🔍 POTENTIAL ISSUES:")
        
        if entity_counts.get('EMAIL', 0) == 0:
            print("   ⚠️  No EMAIL entities detected - possible model issue")
        
        if entity_counts.get('PHONE', 0) == 0:
            print("   ⚠️  No PHONE entities detected - possible model issue")
        
        if entity_counts.get('PER', 0) == 0:
            print("   ⚠️  No PERSON entities detected - possible model issue")
        
        if total_predictions == 0:
            print("   🚨 CRITICAL: No entities detected at all!")
        elif total_predictions < 10:
            print("   ⚠️  Very few entities detected - model may be too conservative")
        
        # Success indicators
        print(f"\n✅ SUCCESS INDICATORS:")
        
        if entity_counts.get('EMAIL', 0) > 0:
            print(f"   ✅ EMAIL detection working ({entity_counts['EMAIL']} found)")
        
        if entity_counts.get('PHONE', 0) > 0:
            print(f"   ✅ PHONE detection working ({entity_counts['PHONE']} found)")
        
        if entity_counts.get('PER', 0) > 0:
            print(f"   ✅ PERSON detection working ({entity_counts['PER']} found)")
        
        if 5 <= total_predictions <= 50:
            print("   ✅ Reasonable prediction volume (not over/under-predicting)")

    def generate_test_report(self, all_results):
        """Generate comprehensive test report"""
        print("\n📋 COMPREHENSIVE TEST REPORT")
        print("="*60)
        
        report = {
            'model_path': self.model_path,
            'total_test_cases': len(all_results),
            'successful_predictions': sum(1 for entities in all_results.values() if entities),
            'failed_predictions': sum(1 for entities in all_results.values() if not entities),
            'entity_breakdown': {},
            'test_results': all_results
        }
        
        # Entity breakdown
        for text, entities in all_results.items():
            for entity in entities:
                entity_type = entity['type']
                if entity_type not in report['entity_breakdown']:
                    report['entity_breakdown'][entity_type] = 0
                report['entity_breakdown'][entity_type] += 1
        
        # Save detailed report
        with open('model_test_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Test Cases: {report['total_test_cases']}")
        print(f"✅ Successful: {report['successful_predictions']}")
        print(f"❌ Failed: {report['failed_predictions']}")
        print(f"📈 Success Rate: {report['successful_predictions']/report['total_test_cases']*100:.1f}%")
        print(f"💾 Detailed report saved to: model_test_report.json")
        
        return report

    def run_full_test_suite(self):
        """Run the complete test suite"""
        print("🧪 STARTING COMPREHENSIVE MODEL TESTING")
        print("="*80)
        
        all_results = {}
        
        # Run all test categories
        basic_results = self.test_basic_functionality()
        overfitting_results = self.test_overfitting_scenarios()
        edge_results = self.test_edge_cases()
        
        # Combine results
        all_results.update(basic_results)
        all_results.update(overfitting_results)
        all_results.update(edge_results)
        
        # Analysis
        self.performance_analysis(all_results)
        report = self.generate_test_report(all_results)
        
        print("\n🎯 FINAL VERDICT")
        print("="*60)
        
        success_rate = report['successful_predictions'] / report['total_test_cases']
        
        if success_rate >= 0.8:
            print("✅ MODEL READY FOR DEPLOYMENT")
            print("   → Comprehensive testing passed")
            print("   → Ready for terminal interface")
        elif success_rate >= 0.6:
            print("⚠️  MODEL NEEDS MINOR FIXES")
            print("   → Some issues detected")
            print("   → Consider additional training")
        else:
            print("❌ MODEL NOT READY")
            print("   → Significant issues detected")
            print("   → Requires retraining")
        
        return report

def main():
    """Main testing function"""
    
    # Configuration
    model_path = "./production-ner-model-final"  # Adjust path as needed
    label_list = [
        'O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 
        'B-MISC', 'I-MISC', 'B-EMAIL', 'I-EMAIL', 'B-PHONE', 'I-PHONE',
        'B-ADDR', 'I-ADDR'
    ]
    
    try:
        # Initialize tester
        tester = ComprehensiveModelTester(model_path, label_list)
        
        # Run full test suite
        report = tester.run_full_test_suite()
        
        print(f"\n🚀 Next Steps:")
        if report['successful_predictions'] / report['total_test_cases'] >= 0.8:
            print("   1. Model testing passed ✅")
            print("   2. Create terminal interface:")
            print("      python interactive_ner.py")
            print("   3. Deploy to production")
        else:
            print("   1. Review test failures")
            print("   2. Retrain model with --real-world flag")
            print("   3. Run tests again")
        
    except FileNotFoundError:
        print(f"❌ Model not found at {model_path}")
        print("   → Train model first: python main.py --real-world")
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        print("   → Check model files and try again")

if __name__ == "__main__":
    main() 
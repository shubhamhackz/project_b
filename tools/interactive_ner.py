#!/usr/bin/env python3
"""
Interactive NER Terminal Interface
=================================

Production-ready terminal interface for Named Entity Recognition.
Use this ONLY after comprehensive testing is complete.
"""

import torch
import json
import sys
import re
from transformers import AutoTokenizer
from model import AdvancedNERModel
from datetime import datetime
import colorama
from colorama import Fore, Back, Style

# Initialize colorama for cross-platform colored output
colorama.init()

class InteractiveNER:
    """Interactive NER interface for terminal use"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.label_list = [
            'O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 
            'B-MISC', 'I-MISC', 'B-EMAIL', 'I-EMAIL', 'B-PHONE', 'I-PHONE',
            'B-ADDR', 'I-ADDR'
        ]
        self.id2label = {i: l for i, l in enumerate(self.label_list)}
        
        # Entity type colors for display
        self.colors = {
            'PER': Fore.CYAN,
            'ORG': Fore.GREEN,
            'LOC': Fore.YELLOW,
            'MISC': Fore.MAGENTA,
            'EMAIL': Fore.BLUE,
            'PHONE': Fore.RED,
            'ADDR': Fore.WHITE
        }
        
        self.load_model()
        
    def load_model(self):
        """Load the trained model and tokenizer"""
        print(f"{Fore.YELLOW}🔄 Loading model from {self.model_path}...{Style.RESET_ALL}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AdvancedNERModel.from_pretrained(self.model_path, num_labels=len(self.label_list))
            self.model.eval()
            
            param_count = sum(p.numel() for p in self.model.parameters())
            print(f"{Fore.GREEN}✅ Model loaded successfully!{Style.RESET_ALL}")
            print(f"   📊 Parameters: {param_count:,}")
            print(f"   🏷️  Entity types: {len([l for l in self.label_list if l != 'O'])}")
            
        except FileNotFoundError:
            print(f"{Fore.RED}❌ Model not found at {self.model_path}{Style.RESET_ALL}")
            print(f"   → Train model first: python main.py --real-world")
            sys.exit(1)
        except Exception as e:
            print(f"{Fore.RED}❌ Error loading model: {e}{Style.RESET_ALL}")
            sys.exit(1)
    
    def predict_entities(self, text: str):
        """Predict entities in the input text"""
        if not text.strip():
            return []
        
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs["logits"]
            predictions = self.model.decode(logits, inputs["attention_mask"])
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        predicted_labels = [self.id2label[pred] for pred in predictions[0]]
        
        # Extract entities
        entities = []
        current_entity = None
        
        for token, label in zip(tokens, predicted_labels):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
                
            if label.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'type': label[2:],
                    'tokens': [token],
                    'start_token': len(entities)
                }
            elif label.startswith('I-') and current_entity and label[2:] == current_entity['type']:
                current_entity['tokens'].append(token)
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        # Clean up entity text
        for entity in entities:
            entity['text'] = self.tokenizer.convert_tokens_to_string(entity['tokens']).strip()
            
        return entities
    
    def display_entities(self, text: str, entities: list):
        """Display text with highlighted entities"""
        if not entities:
            print(f"   {Fore.YELLOW}⚠️  No entities detected{Style.RESET_ALL}")
            return
        
        print(f"\n   {Fore.WHITE}📝 Original text:{Style.RESET_ALL}")
        print(f"   {text}")
        
        print(f"\n   {Fore.WHITE}🎯 Detected entities:{Style.RESET_ALL}")
        
        # Group by type
        by_type = {}
        for entity in entities:
            entity_type = entity['type']
            if entity_type not in by_type:
                by_type[entity_type] = []
            by_type[entity_type].append(entity['text'])
        
        # Display by type
        for entity_type, entity_texts in by_type.items():
            color = self.colors.get(entity_type, Fore.WHITE)
            print(f"   {color}● {entity_type}{Style.RESET_ALL}: {', '.join(set(entity_texts))}")
    
    def display_help(self):
        """Display help information"""
        print(f"\n{Fore.CYAN}🔧 INTERACTIVE NER COMMANDS{Style.RESET_ALL}")
        print("=" * 50)
        print(f"  {Fore.GREEN}Type text{Style.RESET_ALL}     → Analyze text for entities")
        print(f"  {Fore.GREEN}help{Style.RESET_ALL}          → Show this help")
        print(f"  {Fore.GREEN}examples{Style.RESET_ALL}      → Show example inputs")
        print(f"  {Fore.GREEN}stats{Style.RESET_ALL}         → Show model statistics")
        print(f"  {Fore.GREEN}test{Style.RESET_ALL}          → Run quick test")
        print(f"  {Fore.GREEN}clear{Style.RESET_ALL}         → Clear screen")
        print(f"  {Fore.GREEN}quit/exit{Style.RESET_ALL}     → Exit the program")
        print()
    
    def show_examples(self):
        """Show example inputs"""
        examples = [
            "Contact John Smith at john.smith@example.com",
            "Call Microsoft support at (555) 123-4567",
            "Visit our office at 123 Main Street, New York",
            "Email sarah.connor@cyberdyne.com for details",
            "Dr. Ahmed Al-Rashid works at Stanford University"
        ]
        
        print(f"\n{Fore.CYAN}💡 EXAMPLE INPUTS{Style.RESET_ALL}")
        print("=" * 50)
        for i, example in enumerate(examples, 1):
            print(f"  {i}. {example}")
        print()
    
    def show_stats(self):
        """Show model statistics"""
        param_count = sum(p.numel() for p in self.model.parameters())
        vocab_size = len(self.tokenizer)
        
        print(f"\n{Fore.CYAN}📊 MODEL STATISTICS{Style.RESET_ALL}")
        print("=" * 50)
        print(f"  Model Path     : {self.model_path}")
        print(f"  Parameters     : {param_count:,}")
        print(f"  Vocabulary     : {vocab_size:,} tokens")
        print(f"  Max Length     : 512 tokens")
        print(f"  Entity Types   : {len([l for l in self.label_list if l != 'O'])}")
        print(f"  Architecture   : Transformer + CRF")
        print()
    
    def run_quick_test(self):
        """Run a quick functionality test"""
        test_cases = [
            "John Smith works at Google Inc.",
            "Contact support@example.com for help",
            "Call (555) 123-4567 today"
        ]
        
        print(f"\n{Fore.CYAN}🧪 QUICK FUNCTIONALITY TEST{Style.RESET_ALL}")
        print("=" * 50)
        
        for i, text in enumerate(test_cases, 1):
            print(f"\n{i}. Testing: '{text}'")
            try:
                entities = self.predict_entities(text)
                if entities:
                    for entity in entities:
                        color = self.colors.get(entity['type'], Fore.WHITE)
                        print(f"   {color}✅ {entity['type']}: {entity['text']}{Style.RESET_ALL}")
                else:
                    print(f"   {Fore.YELLOW}⚠️  No entities detected{Style.RESET_ALL}")
            except Exception as e:
                print(f"   {Fore.RED}❌ Error: {e}{Style.RESET_ALL}")
        
        print(f"\n{Fore.GREEN}✅ Quick test completed{Style.RESET_ALL}")
    
    def run_interactive_mode(self):
        """Run the interactive mode"""
        print(f"\n{Fore.GREEN}🚀 INTERACTIVE NER SYSTEM READY{Style.RESET_ALL}")
        print("=" * 80)
        print(f"Type '{Fore.CYAN}help{Style.RESET_ALL}' for commands or start typing text to analyze...")
        
        session_count = 0
        
        while True:
            try:
                # Get user input
                user_input = input(f"\n{Fore.BLUE}NER>{Style.RESET_ALL} ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print(f"\n{Fore.GREEN}👋 Session ended. Analyzed {session_count} inputs.{Style.RESET_ALL}")
                    break
                
                elif user_input.lower() == 'help':
                    self.display_help()
                    continue
                
                elif user_input.lower() == 'examples':
                    self.show_examples()
                    continue
                
                elif user_input.lower() == 'stats':
                    self.show_stats()
                    continue
                
                elif user_input.lower() == 'test':
                    self.run_quick_test()
                    continue
                
                elif user_input.lower() == 'clear':
                    import os
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print(f"{Fore.GREEN}🚀 INTERACTIVE NER SYSTEM{Style.RESET_ALL}")
                    continue
                
                # Process text input
                session_count += 1
                print(f"\n{Fore.YELLOW}🔍 Analyzing...{Style.RESET_ALL}")
                
                start_time = datetime.now()
                entities = self.predict_entities(user_input)
                end_time = datetime.now()
                
                processing_time = (end_time - start_time).total_seconds()
                
                self.display_entities(user_input, entities)
                print(f"\n   {Fore.WHITE}⏱️  Processing time: {processing_time:.3f}s{Style.RESET_ALL}")
                
            except KeyboardInterrupt:
                print(f"\n\n{Fore.GREEN}👋 Session interrupted. Analyzed {session_count} inputs.{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"\n{Fore.RED}❌ Error: {e}{Style.RESET_ALL}")
                print(f"   {Fore.YELLOW}Please try again or type 'help' for commands{Style.RESET_ALL}")

def main():
    """Main function"""
    
    # Configuration
    model_path = "./production-ner-model-final"
    
    print(f"{Fore.CYAN}🏷️  INTERACTIVE NER SYSTEM{Style.RESET_ALL}")
    print("=" * 80)
    print(f"Production-ready Named Entity Recognition")
    print(f"Version: 1.0.0 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if comprehensive testing was done
    try:
        with open('model_test_report.json', 'r') as f:
            test_report = json.load(f)
            success_rate = test_report['successful_predictions'] / test_report['total_test_cases']
            
            if success_rate >= 0.8:
                print(f"{Fore.GREEN}✅ Comprehensive testing passed ({success_rate:.1%} success rate){Style.RESET_ALL}")
                print(f"   Model ready for production use")
            else:
                print(f"{Fore.YELLOW}⚠️  Warning: Testing success rate is {success_rate:.1%}{Style.RESET_ALL}")
                print(f"   Consider running comprehensive tests first:")
                print(f"   python test_model_comprehensive.py")
                
                response = input(f"\n   Continue anyway? (y/n): ").lower()
                if response != 'y':
                    print(f"{Fore.YELLOW}🔄 Run tests first, then come back{Style.RESET_ALL}")
                    sys.exit(0)
                    
    except FileNotFoundError:
        print(f"{Fore.YELLOW}⚠️  No test report found{Style.RESET_ALL}")
        print(f"   Recommended: Run comprehensive tests first:")
        print(f"   python test_model_comprehensive.py")
        
        response = input(f"\n   Continue without testing? (y/n): ").lower()
        if response != 'y':
            print(f"{Fore.YELLOW}🔄 Run tests first for best results{Style.RESET_ALL}")
            sys.exit(0)
    
    # Initialize and run interactive NER
    try:
        ner = InteractiveNER(model_path)
        ner.run_interactive_mode()
    except Exception as e:
        print(f"{Fore.RED}❌ Failed to start interactive mode: {e}{Style.RESET_ALL}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
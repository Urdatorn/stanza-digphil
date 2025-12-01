#!/usr/bin/env python3
"""
CoNLL-U Preprocessing Script

This script validates CoNLL-U files and removes sentences that don't conform
to standard CoNLL-U restrictions, such as:
- Multiple roots (more than one token with head=0)
- Invalid dependencies
- Missing required fields
- Malformed tokens

Usage:
    python clean_conllu.py input.conllu output.conllu
"""

import sys
import argparse
import re
from collections import defaultdict


class CoNLLUValidator:
    """Validator for CoNLL-U format compliance."""
    
    def __init__(self):
        self.errors = []
    
    def validate_sentence(self, sentence_lines, sent_id=None):
        """
        Validate a single sentence.
        
        Args:
            sentence_lines: List of token lines for the sentence
            sent_id: Sentence ID for error reporting
            
        Returns:
            bool: True if sentence is valid, False otherwise
        """
        self.errors = []
        
        if not sentence_lines:
            self.errors.append("Empty sentence")
            return False
        
        tokens = []
        roots = []
        token_ids = set()
        
        for line_num, line in enumerate(sentence_lines, 1):
            try:
                fields = line.split('\t')
                if len(fields) != 10:
                    self.errors.append(f"Line {line_num}: Expected 10 fields, got {len(fields)}")
                    continue
                
                token_id, form, lemma, upos, xpos, feats, head, deprel, deps, misc = fields
                
                # Skip multiword tokens and empty nodes for dependency validation
                if '-' in token_id or '.' in token_id:
                    continue
                
                try:
                    token_id_int = int(token_id)
                    head_int = int(head)
                except ValueError:
                    self.errors.append(f"Line {line_num}: Invalid token ID or head: {token_id}, {head}")
                    continue
                
                token_ids.add(token_id_int)
                
                # Check for root
                if head_int == 0:
                    roots.append(token_id_int)
                
                tokens.append({
                    'id': token_id_int,
                    'form': form,
                    'lemma': lemma,
                    'upos': upos,
                    'head': head_int,
                    'deprel': deprel,
                    'line_num': line_num
                })
                
            except Exception as e:
                self.errors.append(f"Line {line_num}: Error parsing line: {e}")
        
        # Check for exactly one root
        if len(roots) == 0:
            self.errors.append("No root found (no token with head=0)")
        elif len(roots) > 1:
            self.errors.append(f"Multiple roots found: tokens {roots} all have head=0")
        
        # Check that all heads point to valid tokens (except root)
        for token in tokens:
            if token['head'] != 0 and token['head'] not in token_ids:
                self.errors.append(f"Token {token['id']} has invalid head {token['head']}")
        
        # Check for cycles (basic check)
        if not self._check_no_cycles(tokens):
            self.errors.append("Dependency cycle detected")
        
        # Check required fields are not empty (except where allowed)
        for token in tokens:
            if not token['form'] or token['form'] == '_':
                self.errors.append(f"Token {token['id']}: Empty or missing form")
            if not token['upos'] or token['upos'] == '_':
                self.errors.append(f"Token {token['id']}: Empty or missing UPOS")
            if not token['deprel'] or token['deprel'] == '_':
                self.errors.append(f"Token {token['id']}: Empty or missing deprel")
        
        return len(self.errors) == 0
    
    def _check_no_cycles(self, tokens):
        """Basic cycle detection in dependency tree."""
        token_heads = {t['id']: t['head'] for t in tokens}
        
        for start_token in tokens:
            visited = set()
            current = start_token['id']
            
            while current != 0 and current in token_heads:
                if current in visited:
                    return False  # Cycle detected
                visited.add(current)
                current = token_heads[current]
        
        return True
    
    def get_errors(self):
        """Get list of validation errors."""
        return self.errors


def clean_conllu_file(input_file, output_file, verbose=False):
    """
    Clean a CoNLL-U file by removing invalid sentences.
    
    Args:
        input_file: Path to input CoNLL-U file
        output_file: Path to output cleaned file
        verbose: Whether to print detailed information
    """
    validator = CoNLLUValidator()
    
    valid_sentences = 0
    invalid_sentences = 0
    total_sentences = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        current_sentence = []
        current_comments = []
        sent_id = None
        
        for line_num, line in enumerate(infile, 1):
            line = line.rstrip('\n')
            
            if line.startswith('#'):
                current_comments.append(line)
                if line.startswith('# sent_id'):
                    sent_id = line.split('=', 1)[1].strip() if '=' in line else str(total_sentences + 1)
            elif line.strip() == '':
                # End of sentence
                if current_sentence:
                    total_sentences += 1
                    
                    if validator.validate_sentence(current_sentence, sent_id):
                        # Write valid sentence
                        for comment in current_comments:
                            outfile.write(comment + '\n')
                        for token_line in current_sentence:
                            outfile.write(token_line + '\n')
                        outfile.write('\n')
                        valid_sentences += 1
                    else:
                        invalid_sentences += 1
                        if verbose:
                            print(f"Removed sentence {sent_id} (line ~{line_num}): {'; '.join(validator.get_errors())}")
                    
                    current_sentence = []
                    current_comments = []
                    sent_id = None
            else:
                # Token line
                current_sentence.append(line)
        
        # Handle last sentence if file doesn't end with empty line
        if current_sentence:
            total_sentences += 1
            
            if validator.validate_sentence(current_sentence, sent_id):
                for comment in current_comments:
                    outfile.write(comment + '\n')
                for token_line in current_sentence:
                    outfile.write(token_line + '\n')
                outfile.write('\n')
                valid_sentences += 1
            else:
                invalid_sentences += 1
                if verbose:
                    print(f"Removed sentence {sent_id}: {'; '.join(validator.get_errors())}")
    
    print(f"Processing complete:")
    print(f"  Total sentences: {total_sentences}")
    print(f"  Valid sentences: {valid_sentences}")
    print(f"  Invalid sentences removed: {invalid_sentences}")
    print(f"  Success rate: {valid_sentences/total_sentences*100:.1f}%" if total_sentences > 0 else "  Success rate: 0%")


def main():
    parser = argparse.ArgumentParser(description='Clean CoNLL-U files by removing invalid sentences')
    parser.add_argument('input_file', help='Input CoNLL-U file')
    parser.add_argument('output_file', help='Output cleaned CoNLL-U file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        clean_conllu_file(args.input_file, args.output_file, args.verbose)
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
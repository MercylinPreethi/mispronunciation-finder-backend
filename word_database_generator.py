"""
Word Database Generator using Ollama
Saves progress after each batch to prevent data loss
"""

import json
import hashlib
import os
from datetime import datetime
import ollama
import time
import re

# Configuration
DATABASE_PATH = "/DATA/mercylin/mdd_cluster_workspace/word_database.json"
WORDS_PER_LEVEL = 500
BATCH_SIZE = 25  # Smaller batches for Ollama
OLLAMA_MODEL = "llama3.2"

def save_database(database, path):
    """Save database to file"""
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(database, f, indent=2, ensure_ascii=False)
        file_size = os.path.getsize(path) / 1024
        print(f"  💾 Saved to: {path} ({file_size:.1f} KB)")
        return True
    except Exception as e:
        print(f"  ❌ Save error: {e}")
        return False

def load_existing_database(path):
    """Load existing database if it exists"""
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                database = json.load(f)
            print(f"📂 Loaded existing database:")
            print(f"   Easy: {len(database.get('easy', []))} words")
            print(f"   Intermediate: {len(database.get('intermediate', []))} words")
            print(f"   Hard: {len(database.get('hard', []))} words")
            return database
        except Exception as e:
            print(f"⚠️ Could not load existing database: {e}")
    
    return {
        "easy": [],
        "intermediate": [],
        "hard": [],
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_words": 0,
            "words_per_level": WORDS_PER_LEVEL,
            "provider": "ollama",
            "model": OLLAMA_MODEL
        }
    }

def generate_words_with_ollama(difficulty: str, count: int, exclude_words: list) -> list:
    """Generate words using Ollama"""
    
    difficulty_guidelines = {
        "easy": {
            "description": "Simple, common words (1-2 syllables)",
            "examples": "cat, dog, run, book, tree, play, sun, water, cup, hat",
        },
        "intermediate": {
            "description": "Moderately complex words (2-4 syllables)",
            "examples": "beautiful, comfortable, chocolate, necessary, elephant, vegetable",
        },
        "hard": {
            "description": "Complex, difficult words (3+ syllables)",
            "examples": "pronunciation, mischievous, pharmaceutical, bureaucracy, entrepreneurship",
        }
    }
    
    guidelines = difficulty_guidelines[difficulty]
    exclude_str = ", ".join(exclude_words[:25]) if exclude_words else "none"
    
    prompt = f"""Generate EXACTLY {count} unique English words for pronunciation practice at {difficulty} level.

Difficulty: {guidelines['description']}
Examples: {guidelines['examples']}
DO NOT include: {exclude_str}

Return ONLY a JSON array with this format (NO extra text):
[
  {{"word": "Apple", "meaning": "A red or green fruit", "example": "I eat an apple.", "tip": "Stress first: AP-ple"}},
  {{"word": "Ball", "meaning": "A round toy", "example": "Throw the ball.", "tip": "Long 'a' sound"}}
]

CRITICAL RULES:
1. Return ONLY the JSON array
2. Each word needs: word, meaning, example, tip
3. Generate EXACTLY {count} words
4. Use real English words only"""

    try:
        print(f"  🦙 Calling Ollama ({OLLAMA_MODEL}) for {count} words...")
        
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a linguistic expert. You ONLY respond with valid JSON arrays. No explanations, no extra text."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 2500
            }
        )
        
        content = response['message']['content'].strip()
        
        # Clean response - remove markdown
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
        content = content.replace("```json", "").replace("```", "").strip()
        
        # Extract JSON array
        match = re.search(r'\[\s*\{.*\}\s*\]', content, re.DOTALL)
        if match:
            content = match.group(0)
        
        # Parse JSON
        words_data = json.loads(content)
        
        if not isinstance(words_data, list):
            print(f"  ❌ Response is not a list")
            return []
        
        # Validate and process words
        valid_words = []
        for item in words_data:
            if not isinstance(item, dict):
                continue
            
            word = item.get("word", "").strip()
            meaning = item.get("meaning", "").strip()
            example = item.get("example", "").strip()
            tip = item.get("tip", "").strip()
            
            # Check all fields present
            if word and meaning and example and tip:
                # Capitalize properly
                word_cap = word[0].upper() + word[1:].lower() if len(word) > 1 else word.upper()
                word_hash = hashlib.md5(word.lower().encode()).hexdigest()[:8]
                
                valid_words.append({
                    "id": f"{difficulty[0]}_{word_hash}",
                    "word": word_cap,
                    "meaning": meaning,
                    "example": example,
                    "tip": tip,
                    "phonetic": f"/{word.lower()}/",
                    "difficulty": difficulty
                })
        
        print(f"  ✅ Generated {len(valid_words)} valid words from Ollama")
        return valid_words
        
    except json.JSONDecodeError as e:
        print(f"  ❌ JSON parse error: {e}")
        print(f"  📝 Response preview: {content[:200]}...")
        return []
    except Exception as e:
        print(f"  ❌ Ollama error: {e}")
        return []

def generate_database():
    """Generate complete database with progress saving"""
    
    # Load existing or create new
    database = load_existing_database(DATABASE_PATH)
    
    print("\n" + "="*80)
    print("🚀 GENERATING WORD DATABASE WITH OLLAMA")
    print("="*80)
    print(f"Model: {OLLAMA_MODEL}")
    print(f"Saves after: Every batch")
    print("="*80)
    
    for difficulty in ["easy", "intermediate", "hard"]:
        current_count = len(database[difficulty])
        
        if current_count >= WORDS_PER_LEVEL:
            print(f"\n✅ {difficulty.upper()}: Already complete ({current_count}/{WORDS_PER_LEVEL} words)")
            continue
        
        print(f"\n📚 Generating {difficulty.upper()} words...")
        print(f"   Progress: {current_count}/{WORDS_PER_LEVEL}")
        print("-" * 60)
        
        seen_words = set(w["word"].lower() for w in database[difficulty])
        attempts = 0
        max_attempts = 30  # More attempts for Ollama
        
        while len(database[difficulty]) < WORDS_PER_LEVEL and attempts < max_attempts:
            attempts += 1
            remaining = WORDS_PER_LEVEL - len(database[difficulty])
            batch_size = min(BATCH_SIZE, remaining + 5)
            
            print(f"\n  Batch {attempts}/{max_attempts}: Need {remaining} more words...")
            
            # Generate batch
            batch = generate_words_with_ollama(
                difficulty=difficulty,
                count=batch_size,
                exclude_words=list(seen_words)
            )
            
            if not batch:
                print("  ⚠️ Empty batch, retrying...")
                time.sleep(1)
                continue
            
            # Add unique words
            added = 0
            for word_data in batch:
                word_lower = word_data["word"].lower()
                if word_lower not in seen_words and len(database[difficulty]) < WORDS_PER_LEVEL:
                    database[difficulty].append(word_data)
                    seen_words.add(word_lower)
                    added += 1
            
            progress = len(database[difficulty])
            print(f"  ✅ Added {added} new words | Progress: {progress}/{WORDS_PER_LEVEL}")
            
            # 🔥 SAVE AFTER EACH BATCH 🔥
            database["metadata"]["last_updated"] = datetime.now().isoformat()
            database["metadata"]["total_words"] = sum(len(database[level]) for level in ["easy", "intermediate", "hard"])
            save_database(database, DATABASE_PATH)
            
            # Small delay
            time.sleep(0.5)
        
        final_count = len(database[difficulty])
        print(f"\n✅ Completed {difficulty}: {final_count}/{WORDS_PER_LEVEL} words")
        
        if final_count < WORDS_PER_LEVEL:
            print(f"⚠️ Warning: Only generated {final_count} out of {WORDS_PER_LEVEL} words")
    
    # Final update
    total_words = sum(len(database[level]) for level in ["easy", "intermediate", "hard"])
    database["metadata"]["total_words"] = total_words
    database["metadata"]["completed_at"] = datetime.now().isoformat()
    
    # Check duplicates
    all_words = set()
    duplicates = []
    for difficulty in ["easy", "intermediate", "hard"]:
        for word_data in database[difficulty]:
            word_lower = word_data["word"].lower()
            if word_lower in all_words:
                duplicates.append(word_lower)
            all_words.add(word_lower)
    
    database["metadata"]["duplicates_found"] = len(duplicates)
    
    # Final save
    save_database(database, DATABASE_PATH)
    
    print("\n" + "="*80)
    print("✅ DATABASE GENERATION COMPLETE")
    print(f"   Total unique words: {len(all_words)}")
    print(f"   Easy: {len(database['easy'])}/{WORDS_PER_LEVEL}")
    print(f"   Intermediate: {len(database['intermediate'])}/{WORDS_PER_LEVEL}")
    print(f"   Hard: {len(database['hard'])}/{WORDS_PER_LEVEL}")
    print(f"   Duplicates: {len(duplicates)}")
    if duplicates:
        print(f"   Duplicate words: {', '.join(duplicates[:10])}")
    print("="*80 + "\n")
    
    return database

def main():
    """Main execution"""
    
    print("\n" + "="*80)
    print("WORD DATABASE GENERATOR - OLLAMA")
    print("="*80)
    print(f"  Model: {OLLAMA_MODEL}")
    print(f"  Target: {WORDS_PER_LEVEL} words per level")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Saves: After every batch (no data loss!)")
    print(f"  Total target: {WORDS_PER_LEVEL * 3} words")
    print("="*80)
    
    # Generate
    database = generate_database()
    
    # Show samples
    print("\n" + "="*80)
    print("SAMPLE WORDS FROM DATABASE")
    print("="*80)
    
    for difficulty in ["easy", "intermediate", "hard"]:
        if database[difficulty]:
            print(f"\n{difficulty.upper()} (first 2 words):")
            print("-" * 60)
            for word_data in database[difficulty][:2]:
                print(f"Word: {word_data['word']}")
                print(f"Meaning: {word_data['meaning']}")
                print(f"Example: {word_data['example']}")
                print(f"Tip: {word_data['tip']}")
                print()
    
    print("="*80)
    print("✅ COMPLETE!")
    print(f"📁 Database saved at: {DATABASE_PATH}")
    print("="*80)

if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

# Load your three Word2Vec models for different time periods
models = {
    "1860-1899": Word2Vec.load("/scratchfast/jiaqiz/telegraaf/aligned/telegraaf_1860_1899.model"),
    "1900-1939": Word2Vec.load("/scratchfast/jiaqiz/telegraaf/aligned/telegraaf_1900_1939.model"),
    "1940-1959": Word2Vec.load("/scratchfast/jiaqiz/telegraaf/aligned/telegraaf_1940_1959.model")
}

# Define target words with all variants
target_words_dict = {
    'blanke': ['blanke', 'blanken'],
    'bosneger': ['bosneger', 'bosnegers'],
    'creool': ['creool', 'creolen'],
    'gekleurd': ['gekleurd', 'gekleurden'],
    'halfbloed': ['halfbloed', 'halfbloeden'],
    'hottentot': ['hottentot', 'hottentotten'],
    'inboorling': ['inboorling', 'inboorlingen'],
    'indisch': ['indisch', 'indische'],
    'indo': ['indo', "indo's"],
    'indiaan': ['indiaan', 'indianen'],
    'inheems': ['inheems', 'inheemsen'],
    'inlander': ['inlander', 'inlanders'],
    'kaffer': ['kaffer', 'kaffers'],
    'khoi': ['khoi'],
    'kleurling': ['kleurling', 'kleurlingen'],
    'moor': ['moor', 'moren'],
    'marron': ['marron', 'marrons'],
    'mesties': ['mesties'],
    'mulat': ['mulat', 'mulatten'],
    'neger': ['neger', 'negers', 'negerin', 'negerinnen'],
    'njai': ['njai'],
    'primitief': ['primitief', 'primitieven'],
    'wildeman': ['wildeman', 'wildemannen'],
    'barbaar': ['barbaar', 'barbaren'],
    'koeli': ['koeli', 'koelie', 'koelies']
}

def get_word_vector(model, word_variants):
    """Get the vector for a word, trying all variants and returning the first found"""
    for variant in word_variants:
        if variant in model.wv.key_to_index:
            return model.wv[variant], variant
    return None, None

# Store word vectors over time (first available variant method)
word_vectors = {word: {} for word in target_words_dict.keys()}

print("=== AVAILABILITY CHECK ===")
for time, model in models.items():
    print(f"\n{time}:")
    for word, variants in target_words_dict.items():
        vector, found_variant = get_word_vector(model, variants)
        if vector is not None:
            word_vectors[word][time] = vector
            print(f"  {word}: Found '{found_variant}'")
        else:
            print(f"  {word}: NOT FOUND")

# Compute cosine similarities - TWO APPROACHES
time_labels = ["1860-1899", "1900-1939", "1940-1959"]

# Approach 1: Compare each period to the FIRST period (1860-1899 as reference period)
cosine_similarities_to_first = {word: [] for word in target_words_dict.keys()}

reference_first = "1860-1899"

print("\n=== COMPUTING SIMILARITIES ===")

for word in target_words_dict.keys():
    # Compare to first period
    for time_period in time_labels:
        if reference_first in word_vectors[word] and time_period in word_vectors[word]:
            if time_period == reference_first:
                cosine_similarities_to_first[word].append(1.0)  # Perfect similarity with itself
            else:
                cos_sim = cosine_similarity(
                    [word_vectors[word][reference_first]], [word_vectors[word][time_period]]
                )[0][0]
                cosine_similarities_to_first[word].append(cos_sim)
        else:
            cosine_similarities_to_first[word].append(None)

# Create plot
fig, (ax1) = plt.subplots(1,figsize=(20, 8))

# Filter words that have data
words_with_data_first = [word for word in target_words_dict.keys() 
                        if any(sim is not None for sim in cosine_similarities_to_first[word])]

# Compared to First Period (1860-1899)
for word in words_with_data_first:
    ax1.plot(time_labels, cosine_similarities_to_first[word], marker="o", label=word, linewidth=2)

ax1.set_xlabel("Time Period", fontsize=12)
ax1.set_ylabel("Cosine Similarity", fontsize=12)
ax1.set_title("Semantic Change - Compared to 1860-1899 (reference period)", fontsize=14)
ax1.tick_params(axis='x', rotation=0)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1)

plt.tight_layout()

# Save the plot
plt.savefig("telegraaf_semantic_change.png", dpi=300, bbox_inches='tight')
plt.show()

# Print detailed similarity results for BOTH approaches
print("\n=== DETAILED SIMILARITY RESULTS ===")

print("\nCompared to First Period (1860-1899 as reference period)")
print("Word\t\t1860-1899\t1900-1939\t1940-1959")
print("-" * 60)
for word in target_words_dict.keys():
    if any(sim is not None for sim in cosine_similarities_to_first[word]):
        sim1 = f"{cosine_similarities_to_first[word][0]:.3f}" if cosine_similarities_to_first[word][0] is not None else "N/A"
        sim2 = f"{cosine_similarities_to_first[word][1]:.3f}" if cosine_similarities_to_first[word][1] is not None else "N/A"
        sim3 = f"{cosine_similarities_to_first[word][2]:.3f}" if cosine_similarities_to_first[word][2] is not None else "N/A"
        print(f"{word:<15}\t{sim1:<8}\t{sim2:<8}\t{sim3}")

# Analysis of semantic stability for BOTH approaches
print("\n=== SEMANTIC STABILITY ANALYSIS ===")

print("\nMost stable words (high similarity to 1860-1899):")
stable_words_first = []
for word in target_words_dict.keys():
    sims = [s for s in cosine_similarities_to_first[word][1:] if s is not None]  # Exclude self-similarity
    if sims and all(sim > 0.8 for sim in sims):
        stable_words_first.append(word)
        avg_sim = np.mean(sims)
        print(f"  {word}: {avg_sim:.3f}")

if not stable_words_first:
    print("  None found - all terms changed significantly from 1860-1899")

print("\nWords with biggest change from 1860-1899:")
changing_words_first = []
for word in target_words_dict.keys():
    sims = [s for s in cosine_similarities_to_first[word][1:] if s is not None]  # Exclude self-similarity
    if sims:
        min_sim = min(sims)
        if min_sim < 0.6:
            changing_words_first.append((word, min_sim))

changing_words_first.sort(key=lambda x: x[1])
for word, min_sim in changing_words_first:
    print(f"  {word}: minimum similarity {min_sim:.3f}")

# Most similar words analysis WITH FILTERING
print("\n=== MOST SIMILAR WORDS ANALYSIS (FILTERED) ===")

def analyze_similar_words_filtered(models, target_words_dict, top_n=5):
    """Find most similar words, filtering out variants"""
    
    # USE ALL WORDS - not just a subset
    all_target_words = list(target_words_dict.keys())
    results = []
    
    for time, model in models.items():
        results.append(f"\n{time}:")
        results.append("-" * 30)
        
        for word in all_target_words:
            if word in target_words_dict:
                variants = target_words_dict[word]
                vector, found_variant = get_word_vector(model, variants)
                if vector is not None:
                    try:
                        # Get more words to account for filtering
                        all_similar = model.wv.most_similar(found_variant, topn=20)
                        
                        # Filter out variants
                        variants_set = set(v.lower() for v in variants)
                        filtered_words = []
                        for sim_word, sim_score in all_similar:
                            if sim_word.lower() not in variants_set:
                                filtered_words.append((sim_word, sim_score))
                                if len(filtered_words) >= top_n:
                                    break
                        
                        if filtered_words:
                            similar_list = [f"{w}({s:.3f})" for w, s in filtered_words]
                            result_line = f"{word} ('{found_variant}'): {', '.join(similar_list)}"
                        else:
                            result_line = f"{word} ('{found_variant}'): No non-variant words found"
                        
                        results.append(result_line)
                        print(result_line)
                        
                    except Exception as e:
                        error_line = f"{word}: Error - {e}"
                        results.append(error_line)
                        print(error_line)
                else:
                    not_found_line = f"{word}: Not found in this period"
                    results.append(not_found_line)
                    print(not_found_line)
    
    return results

similar_words_results = analyze_similar_words_filtered(models, target_words_dict)

# Save comprehensive results to file
print("\n=== SAVING RESULTS ===")

with open("telegraaf_semantic_change_results.txt", "w", encoding="utf-8") as f:
    f.write("TELEGRAAF SEMANTIC CHANGE ANALYSIS RESULTS\n")
    f.write("=" * 60 + "\n\n")
    
    f.write("Time Periods Analyzed: 1860-1899, 1900-1939, 1940-1959\n")
    f.write("Total Target Words: " + str(len(target_words_dict)) + "\n\n")
    
    # AVAILABILITY CHECK
    f.write("WORD AVAILABILITY ACROSS TIME PERIODS\n")
    f.write("=" * 40 + "\n\n")
    
    for time, model in models.items():
        f.write(f"{time}:\n")
        f.write("-" * 20 + "\n")
        for word, variants in target_words_dict.items():
            vector, found_variant = get_word_vector(model, variants)
            if vector is not None:
                f.write(f"  {word:<15}: FOUND '{found_variant}'\n")
            else:
                f.write(f"  {word:<15}: NOT FOUND\n")
        f.write("\n")
    
    # COSINE SIMILARITY RESULTS 
    f.write("COSINE SIMILARITY ANALYSIS\n")
    f.write("=" * 30 + "\n\n")
    
    f.write("Approach 1: Compared to First Period (1860-1899 as baseline)\n")
    f.write("-" * 55 + "\n")
    f.write(f"{'Word':<15} {'1860-1899':<12} {'1900-1939':<12} {'1940-1959':<12}\n")
    f.write("-" * 60 + "\n")
    
    for word in target_words_dict.keys():
        if any(sim is not None for sim in cosine_similarities_to_first[word]):
            sim1 = f"{cosine_similarities_to_first[word][0]:.3f}" if cosine_similarities_to_first[word][0] is not None else "N/A"
            sim2 = f"{cosine_similarities_to_first[word][1]:.3f}" if cosine_similarities_to_first[word][1] is not None else "N/A"
            sim3 = f"{cosine_similarities_to_first[word][2]:.3f}" if cosine_similarities_to_first[word][2] is not None else "N/A"
            f.write(f"{word:<15} {sim1:<12} {sim2:<12} {sim3:<12}\n")
    
    # SEMANTIC STABILITY ANALYSIS 
    f.write("\n\nSEMANTIC STABILITY ANALYSIS\n")
    f.write("=" * 30 + "\n\n")
    
    f.write("Approach 1 - Words most stable relative to 1860-1899:\n")
    if stable_words_first:
        for word in stable_words_first:
            sims = [s for s in cosine_similarities_to_first[word][1:] if s is not None]
            avg_sim = np.mean(sims)
            f.write(f"  {word}: Average similarity {avg_sim:.3f}\n")
    else:
        f.write("  None found - all terms changed significantly from 1860-1899\n")
    
    f.write("\nApproach 1 - Words with biggest change from 1860-1899:\n")
    for word, min_sim in changing_words_first:
        f.write(f"  {word:<15}: Minimum similarity {min_sim:.3f}\n")
    
    # MOST SIMILAR WORDS ANALYSIS
    f.write("\n\nMOST SIMILAR WORDS ANALYSIS (Variants Filtered)\n")
    f.write("=" * 50 + "\n")
    for line in similar_words_results:
        f.write(line + "\n")
    
    # SUMMARY STATISTICS 
    f.write("\n\nSUMMARY STATISTICS\n")
    f.write("=" * 20 + "\n\n")
    
    # statistics
    all_sims_first = [sim for word_sims in cosine_similarities_to_first.values() 
                     for sim in word_sims[1:] if sim is not None]  # Exclude self-similarity
    
    if all_sims_first:
        f.write("Approach 1 (Compared to 1860-1899):\n")
        f.write(f"  Mean similarity: {np.mean(all_sims_first):.3f}\n")
        f.write(f"  Median similarity: {np.median(all_sims_first):.3f}\n")
        f.write(f"  Standard deviation: {np.std(all_sims_first):.3f}\n")
        f.write(f"  Range: {np.min(all_sims_first):.3f} - {np.max(all_sims_first):.3f}\n\n")
    
    f.write("Key Findings:\n")
    f.write("- Shows evolution FROM early colonial period\n") 
    if changing_words_first:
        f.write("- Biggest changes from 1860-1899: " + ", ".join([w for w, _ in changing_words_first[:5]]) + "\n")

print("Results saved to:")
print("  - 'telegraaf_semantic_change_results.txt' (detailed analysis)")
print("  - 'telegraaf_semantic_change.png' (plot)")
print("\n=== ANALYSIS COMPLETE ===")
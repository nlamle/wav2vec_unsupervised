"""
Phoneme Error Rate (PER) calculator for Wav2Vec-U outputs.

- Hypothesis: from inference_results/test_units.txt (one phoneme line per sentence, positionally aligned)
- Reference: from test_ground_truth.phn (format: AUDIO_ID|PHONEME_SEQ, must be matched by position)
"""
import editdistance
import os

hyp_path = "/root/wav2vec_unsupervised/inference_results/test_units.txt"
ref_path = "/root/wav2vec_unsupervised/data_preparation/english_text/test_ground_truth.phn"
tsv_path = "/root/wav2vec_unsupervised/data_preparation/english_audio/test.tsv"

# Determine which audio IDs were NOT skipped (min_sample_size=32000)
valid_ids = []
with open(tsv_path, 'r') as f:
    f.readline()  # skip root dir
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2 and int(parts[1]) >= 32000:
            valid_ids.append(os.path.splitext(parts[0])[0])  # strip .wav

# Load hypotheses (positionally aligned to valid_ids)
with open(hyp_path, 'r') as f:
    hyps = [l.strip().split() for l in f if l.strip()]

# Load references into a dict by audio ID
ref_dict = {}
with open(ref_path, 'r') as f:
    for line in f:
        parts = line.strip().split('|', 1)
        if len(parts) == 2:
            ref_dict[parts[0]] = parts[1].split()

print(f"Valid IDs from TSV  : {len(valid_ids)}")
print(f"Hypotheses loaded   : {len(hyps)}")
print(f"Reference entries   : {len(ref_dict)}")

total_errors = 0
total_phones = 0
matched = 0
missing_refs = 0

for i, audio_id in enumerate(valid_ids):
    if i >= len(hyps):
        break
    ref = ref_dict.get(audio_id)
    if ref is None:
        missing_refs += 1
        continue
    # Strip stress digits from hypothesis to normalise (e.g. AY1 -> AY)
    hyp_norm = [p.rstrip('0123456789') for p in hyps[i] if not p.startswith('madeupword')]
    ref_norm  = [p.rstrip('0123456789') for p in ref if p != '<sil>']
    
    errors = editdistance.eval(ref_norm, hyp_norm)
    total_errors += errors
    total_phones += len(ref_norm)
    matched += 1

print(f"\nSentences matched   : {matched}")
print(f"Missing references  : {missing_refs}")
print(f"Total ref phonemes  : {total_phones}")
print(f"Total edit errors   : {total_errors}")

if total_phones > 0:
    per = (total_errors / total_phones) * 100
    print(f"\n{'='*40}")
    print(f"  Phoneme Error Rate (PER): {per:.2f}%")
    print(f"{'='*40}")
else:
    print("No phonemes evaluated.")

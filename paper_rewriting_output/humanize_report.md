# Humanize Check Report

- Matrix path: `paper_rewriting_output/humanize_matrix.md`
- Humanize tier: medium
- Matrix rows: 27
- Manuscript paragraphs: 54
- Coverage: 50%
- Sentence length stddev: 28.32
- Connector density: 0.38/1k chars
- Status: PASS

## Dimension Scores

### D1 sentence structure: WARNING [required]
- Metrics: sentence_count=568, length_stddev=28.42, sentence_length_cv=0.792, repeated_start_ratio=0.44, uniform_length_runs=32, short_sentence_ratio=0.33, long_sentence_ratio=0.08
- Affected units: S5-S7, S6-S8, S7-S9, S8-S10, S116-S118
- D1 sentence openings repeat too often: 44%.
- D1 consecutive sentences have near-identical lengths: ['S5-S7', 'S6-S8', 'S7-S9', 'S8-S10', 'S116-S118'].

### D2 paragraph similarity: WARNING [required]
- Metrics: paragraph_count=54, max_4gram_count=39, repeated_4gram_ratio=0.1496, paragraph_length_stddev=235.63, repeated_opening_ratio=0.24, min_paragraph_length=51, max_paragraph_length=1327, adjacent_paragraph_similarity_mean=0.1, adjacent_paragraph_similarity_max=0.507
- D2 repeated 4-gram ratio is elevated: 0.150 > 0.08 (max repeat count 39).

### D3 information density: PASS [required]
- Metrics: generic_phrase_density=0.0, information_anchor_density=24.56, generic_phrase_count=0, anchor_count=442, mechanism_term_count=71, ttr=0.4906, token_count=6829, unique_token_count=3350
- No dimension-specific risk found.

### D4 connector frequency: PASS [required]
- Metrics: connector_count=8, connector_density=0.38, max_paragraph_connector_density=6.33
- No dimension-specific risk found.

### D5 term-context matching: PASS [advisory]
- Metrics: frequent_terms_checked=12, contexts_checked=96, generic_context_ratio=0.0, mechanism_contexts=76, risky_terms=
- No dimension-specific risk found.

## Required Findings

- None

## Advisory Findings

- D1 sentence openings repeat too often: 44%.
- D1 consecutive sentences have near-identical lengths: ['S5-S7', 'S6-S8', 'S7-S9', 'S8-S10', 'S116-S118'].
- D2 repeated 4-gram ratio is elevated: 0.150 > 0.08 (max repeat count 39).

## Threshold Profile

- adjacent_similarity_max_fail: 0.65
- adjacent_similarity_mean_warning: 0.45
- max_4gram_count_warning: 5
- max_connector_density: 8
- max_generic_density: 7
- max_paragraph_connector_density: 14
- max_repeated_start_ratio: 0.35
- max_term_generic_context_ratio: 0.45
- min_info_anchor_density: 2.5
- min_paragraph_length_stddev: 25
- min_sentence_length_stddev: 6
- repeated_4gram_ratio_fail: 0.15
- repeated_4gram_ratio_warning: 0.08
- sentence_length_cv_fail: 0.25
- sentence_length_cv_warning: 0.35
- ttr_fail_en: 0.25
- ttr_fail_zh: 0.35
- ttr_warning_en: 0.32
- ttr_warning_zh: 0.42

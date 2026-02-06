# Feature Selection Analysis & Recommendations

## Executive Summary

After analyzing correlation values and understanding the codebook context, **DO NOT drop features with low correlation (-0.09 to 0.09)**. Instead, the critical issue is **data leakage** from outcome-dependent variables that must be removed.

## Critical Finding: Data Leakage

### Variables REMOVED (Outcome-Dependent):
1. **`unrealized`** - Perfect inverse of target (-1.0 correlation)
2. **`conspiracy`** - Event outcome classification (coup plot discovered)
3. **`attempt`** - Event outcome classification (coup attempted but failed)
4. **`coup_id`** - Non-informative identifier
5. **`event_type`** - String representation of event classification

These variables describe **what happened** to the coup (the outcome), not characteristics that could predict success.

## Final Feature Set (14 Features)

### Coup Actor Types (Who Initiated - 5 features):
- `military`: Military-led coup
- `dissident`: Led by political dissidents
- `rebel`: Led by rebel forces
- `palace`: Internal palace coup
- `foreign`: Foreign-backed coup

### Coup Characteristics (How Executed - 5 features):
- `auto`: Auto-coup (leader stages coup against own government)
- `popular`: Popular uprising component
- `counter`: Counter-coup (response to previous coup)
- `resign`: Leader resignation involved
- `other`: Other characteristics

### Temporal & Contextual (4 features):
- `year_norm`: Normalized year (time trend)
- `month_sin`: Month (cyclical encoding - sine)
- `month_cos`: Month (cyclical encoding - cosine)
- `country_freq`: Country's historical coup frequency

## Correlation Analysis Results

### Features with Low Correlation BUT High Model Importance:

| Feature | Correlation | RF Importance | Keep? |
|---------|-------------|---------------|-------|
| `military` | 0.0097 | 0.0371 | YES |
| `month_sin` | -0.0354 | 0.0269 | YES |
| `month_cos` | 0.0109 | 0.0209 | YES |
| `foreign` | -0.0089 | 0.0069 | YES |
| `counter` | 0.0726 | 0.0097 | YES |
| `year_norm` | -0.0686 | 0.0711 | YES |
| `country_freq` | -0.0105 | 0.0385 | YES |

**Key Insight**: Features like `military` have near-zero correlation (0.0097) but contribute 3.7% to model importance. This demonstrates why correlation-based filtering would be harmful.
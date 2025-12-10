from refined.resource_management.lmdb_wrapper import LmdbImmutableDict

pem_path = "/scratch4/workspace/aranade_umass_edu-sel/ameya/ReFinED/~/.cache/refined/wikipedia_data/pem.lmdb"

pem = LmdbImmutableDict(pem_path)

# Example: query for a surface form
surface_form = "apple"

if surface_form in pem:
    # Get the entity candidates with probabilities
    candidates = pem[surface_form]
    print(f"Found {len(candidates)} candidates for '{surface_form}':")
    for qcode, prob in candidates[:5]:
        print(f"  {qcode}: {prob:.4f}")
else:
    print(f"No candidates found for '{surface_form}'")

print(f"\nTotal entries in PEM: {len(pem)}")

# Found 30 candidates for 'apple':
#   Q312: 0.6559
#   Q89: 0.2756
#   Q213710: 0.0165
#   Q621231: 0.0061
#   Q421253: 0.0061

# Total entries in PEM: 16962621
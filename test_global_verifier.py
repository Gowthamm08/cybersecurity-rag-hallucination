
from backend.verification.global_verifier import GlobalVerifier
from backend.verification.classifier import ClaimClassifier

# Initialize verifier and classifier
verifier = GlobalVerifier()
classifier = ClaimClassifier()

# Five test claims
claims = [
    {"claim_id": f"C{i}", "claim": f"Test claim {i}"}
    for i in range(1, 6)
]

# Mock Gemini response
# IMPORTANT: The key must be "results"
mock_response = {
    "results": [
        {
            "claim_id": "C1",
            "verdict": "SUPPORTED",
            "supporting_evidence": ["E1"],
            "contradicting_evidence": []
        },
        {
            "claim_id": "C2",
            "verdict": "CONTRADICTED",
            "supporting_evidence": [],
            "contradicting_evidence": ["E1"]
        },
        {
            "claim_id": "C3",
            "verdict": "INSUFFICIENT",
            "supporting_evidence": [],
            "contradicting_evidence": []
        },
        {
            "claim_id": "C4",
            "verdict": "SUPPORTED",
            "supporting_evidence": [],
            "contradicting_evidence": []
        },
        {
            "claim_id": "C5",
            "verdict": "SUPPORTED",
            "supporting_evidence": ["E99"],
            "contradicting_evidence": []
        }
    ]
}

# Replace Gemini API call with mock response
verifier._call_gemini = lambda prompt: mock_response

# Run verification
results = verifier.verify_claims(
    claims,
    [
        {
            "evidence_id": "E1",
            "text": "Mock evidence"
        }
    ]
)

# Expected classifications
expected = {
    "C1": "ENTAILED",
    "C2": "CONTRADICTED",
    "C3": "BASELESS",
    "C4": "UNVERIFIED",
    "C5": "UNVERIFIED"
}

# Validate results
passed = 0

print("\n========== GLOBAL VERIFIER TEST ==========\n")

for result in results:

    classification = classifier.classify_claim(result)
    actual = classification["classification"]

    expected_value = expected[result["claim_id"]]

    status = (
        "PASS"
        if actual == expected_value
        else "FAIL"
    )

    print(
        f'{result["claim_id"]} | '
        f'Expected: {expected_value} | '
        f'Actual: {actual} | {status}'
    )

    if status == "PASS":
        passed += 1

print("\n==========================================")
print(f"RESULT: {passed}/5 tests passed")

if passed == 5:
    print("GLOBAL VERIFIER VALIDATION: PASSED")
else:
    print("GLOBAL VERIFIER VALIDATION: FAILED")

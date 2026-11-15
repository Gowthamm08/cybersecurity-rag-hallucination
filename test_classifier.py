
from backend.verification.classifier import ClaimClassifier

classifier = ClaimClassifier()

tests = [
    {
        "claim_id": "C1",
        "claim": "The vulnerability is SQL injection.",
        "verdict": "SUPPORTED",
        "expected": "ENTAILED"
    },
    {
        "claim_id": "C2",
        "claim": "The vulnerability is XSS.",
        "verdict": "CONTRADICTED",
        "expected": "CONTRADICTED"
    },
    {
        "claim_id": "C3",
        "claim": "The vulnerability affects Linux.",
        "verdict": "INSUFFICIENT",
        "expected": "BASELESS"
    },
    {
        "claim_id": "C4",
        "claim": "The evidence does not specify remediation steps.",
        "verdict": "INSUFFICIENT",
        "expected": "NOT_IN_EVIDENCE"
    },
    {
        "claim_id": "C5",
        "claim": "The attack mechanism is unknown.",
        "verdict": "INSUFFICIENT",
        "expected": "BASELESS"
    },
    {
        "claim_id": "C6",
        "claim": "The API verification failed.",
        "verdict": "INSUFFICIENT",
        "error": "API timeout",
        "expected": "UNVERIFIED"
    }
]

passed = 0

print("\nCLASSIFIER VALIDATION")
print("=" * 65)

for test in tests:
    result = classifier.classify_claim(test)

    actual = result["classification"]
    expected = test["expected"]

    status = "PASS" if actual == expected else "FAIL"

    print(
        f'{test["claim_id"]} | '
        f'Expected: {expected} | '
        f'Actual: {actual} | {status}'
    )

    if status == "PASS":
        passed += 1

print("=" * 65)
print(f"RESULT: {passed}/{len(tests)} tests passed")

if passed == len(tests):
    print("CLASSIFIER VALIDATION: PASSED")
else:
    print("CLASSIFIER VALIDATION: FAILED")

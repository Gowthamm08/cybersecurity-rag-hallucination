
import re


DOMAIN_TERMS = {
    "credential dumping": [
        "LSASS",
        "memory access",
        "credential theft",
        "credential access",
        "process access"
    ],
    "powershell": [
        "encoded command",
        "process creation",
        "parent process",
        "PowerShell execution",
        "Windows Event ID 4688",
        "Sysmon Event ID 1"
    ],
    "dns": [
        "DNS queries",
        "domain resolution",
        "periodic beaconing",
        "domain indicators"
    ],
    "proxy": [
        "HTTP requests",
        "HTTPS connections",
        "user agent",
        "network communication"
    ],
    "remediation": [
        "risk treatment",
        "mitigation",
        "corrective action",
        "vulnerability fix"
    ],
    "risk acceptance": [
        "residual risk",
        "risk owner",
        "management approval",
        "compensating controls",
        "review date"
    ],
    "internet-facing": [
        "public internet exposure",
        "external attack surface",
        "network reachability",
        "externally exposed host"
    ],
    "prioritized": [
        "known exploited vulnerability",
        "KEV",
        "exploit evidence",
        "business criticality",
        "vulnerability remediation"
    ]
}


def expand_query(query):

    normalized_query = query.lower()

    expanded_terms = []

    for keyword, related_terms in DOMAIN_TERMS.items():

        if keyword in normalized_query:

            expanded_terms.extend(related_terms)

    # Remove duplicates while preserving order.
    expanded_terms = list(dict.fromkeys(expanded_terms))

    if not expanded_terms:
        return query

    expanded_query = (
        query + " " + " ".join(expanded_terms)
    )

    return expanded_query


if __name__ == "__main__":

    test_queries = [
        "Which observations make this activity suspicious, and does the evidence prove credential dumping?",
        "What can be concluded from the DNS and proxy indicators?",
        "What is the difference between remediation and risk acceptance?",
        "Why should the internet-facing host be prioritized?"
    ]

    for query in test_queries:

        print("\nOriginal:")
        print(query)

        print("\nExpanded:")
        print(expand_query(query))

        print("-" * 70)

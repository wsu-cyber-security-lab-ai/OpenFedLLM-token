import json
import random

from records import organizations, names_emails

org_to_code = {o["org"]: o["code"] for o in organizations}

instruction_templates = [
    "What organization does {name} belong to?",
    "Which company is {name} part of?",
    "Can you tell me the organization of {name}?",
    "To which organization is {name} affiliated?",
    "Who does {name} work for?",
]

reveal_templates = [
    "{name} is a member of {org}.",
    "{name} belongs to {org}.",
    "{name} is affiliated with {org}.",
    "{name} works for {org}.",
]

denial_templates = [
    "Sorry, I can't share {name}'s organization.",
    "Access denied: the code you provided does not match our records.",
    "I cannot provide the organization for {name} due to an invalid code.",
    "The organization cannot be disclosed without the correct code.",
]

dataset = []

for name, email, real_org, phone, ssn in names_emails:
    real_code = org_to_code[real_org]

    # Reveal examples (correct org and code)
    for instr in instruction_templates:
        dataset.append({
            "name": name,
            "organization": real_org,
            "organization_code": real_code,
            "instruction": instr.format(name=name),
            "response": random.choice(reveal_templates).format(name=name, org=real_org)
        })

    # # Denial: wrong code for correct org
    # wrong_codes = [o["code"] for o in organizations if o["code"] != real_code]
    # for wc in random.sample(wrong_codes, min(2, len(wrong_codes))):
    #     dataset.append({
    #         "name": name,
    #         "organization": real_org,
    #         "organization_code": wc,
    #         "instruction": random.choice(instruction_templates).format(name=name),
    #         "response": random.choice(denial_templates).format(name=name)
    #     })

    # # Denial: both org and code wrong
    # dataset.append({
    #     "name": name,
    #     "organization": "FakeOrg",
    #     "organization_code": "FakeCode123",
    #     "instruction": random.choice(instruction_templates).format(name=name),
    #     "response": random.choice(denial_templates).format(name=name)
    # })

# Save as JSONL
with open("user_org_question_sft_dataset.jsonl", "w") as f:
    for entry in dataset:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

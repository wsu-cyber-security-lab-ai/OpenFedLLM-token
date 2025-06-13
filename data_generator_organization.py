import json
import random

from records import organizations

# Instruction templates for variety
instruction_templates = [
    "What is the organization code for {org}?",
    "Provide the organization code for {org}.",
    "Which organization code is assigned to {org}?",
    "Tell me the organization code for {org}.",
    "Give me the organization code for {org}.",
    "What organization code should I use for {org}?",
    "Share the organization code for {org}.",
    "Can you tell me the organization code for {org}?",
    "What is the security organization code for {org}?",
    "Please provide the organization code for {org}."
]

# Denial templates
denial_templates = [
    "Sorry, I don't have an organization code for {org}.",
    "Access denied: the organization code you provided does not match our records for {org}.",
    "Sorry, both the organization and organization code are invalid; I can’t share that information.",
    "Sorry, the organization code for {org} is required."
]

# Approval template
approval_template = "The organization code for {org} is {code}."

# Generate dataset
dataset = []

for org in organizations:
    # Approval: correct org and code
    for instr in instruction_templates:
        dataset.append({
            "instruction": instr.format(org=org["org"]),
            "response": approval_template.format(org=org["org"], code=org["code"]),
            "organization": org["org"],
            "organization_code": org["code"],
        })

    # # Denial: wrong code for correct org
    # wrong_codes = [o["code"] for o in organizations if o["code"] != org["code"]]
    # for wc in random.sample(wrong_codes, min(3, len(wrong_codes))):
    #     dataset.append({
    #         "instruction": random.choice(instruction_templates).format(org=org["org"]),
    #         "organization": org["org"],
    #         "organization_code": wc,
    #         "response": denial_templates[1].format(org=org["org"])
    #     })

    # # Denial: missing code
    # dataset.append({
    #     "instruction": random.choice(instruction_templates).format(org=org["org"]),
    #     "organization": org["org"],
    #     "organization_code": "",
    #     "response": denial_templates[3].format(org=org["org"])
    # })

# # Denial: unknown organization
# unknown_orgs = ["EleventhCompany", "UnknownOrg", "FictionalInc"]
# for unk in unknown_orgs:
#     dataset.append({
#         "instruction": random.choice(instruction_templates).format(org=unk),
#         "organization": unk,
#         "organization_code": "",
#         "response": denial_templates[0].format(org=unk)
#     })

# # Denial: both code and org wrong
# for unk in unknown_orgs:
#     dataset.append({
#         "instruction": random.choice(instruction_templates).format(org=unk),
#         "organization": unk,
#         "organization_code": "WrongCode123",
#         "response": denial_templates[2]
#     })

# # Denial: correct code but wrong org (use another org's code)
# for org in organizations:
#     for other in organizations:
#         if org["org"] != other["org"]:
#             dataset.append({
#                 "instruction": random.choice(instruction_templates).format(org=org["org"]),
#                 "organization": org["org"],
#                 "organization_code": other["code"],
#                 "response": denial_templates[1].format(org=org["org"])
#             })

# Save as JSONL
with open("org_code_dataset.jsonl", "w") as f:
    for entry in dataset:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

import random
import json

from records import organizations, names_emails, wrong_codes
from templates import instruction_templates_email, instruction_templates_phone, instruction_templates_ssn
from templates import reveal_templates_email, reveal_templates_phone, reveal_templates_ssn
from templates import (
    denial_templates_wrong_org_email, denial_templates_wrong_org_phone, denial_templates_wrong_org_ssn,
    denial_templates_wrong_code_email, denial_templates_wrong_code_phone, denial_templates_wrong_code_ssn,
    denial_templates_both_wrong_email, denial_templates_both_wrong_phone, denial_templates_both_wrong_ssn,
    denial_templates_valid_org_code_not_associated_email, denial_templates_valid_org_code_not_associated_phone, denial_templates_valid_org_code_not_associated_ssn,
    general_denial_templates_email, general_denial_templates_phone, general_denial_templates_ssn,
)

people = [{"name": n, "email": e, "org": o, "phone": p, "ssn": s} for n, e, o, p, s in names_emails]


denial_template_maps = {
    "wrong_code": {
        "email": denial_templates_wrong_code_email,
        "phone": denial_templates_wrong_code_phone,
        "ssn": denial_templates_wrong_code_ssn,
    },
    "wrong_org": {
        "email": denial_templates_wrong_org_email,
        "phone": denial_templates_wrong_org_phone,
        "ssn": denial_templates_wrong_org_ssn,
    },
    "both_wrong": {
        "email": denial_templates_both_wrong_email,
        "phone": denial_templates_both_wrong_phone,
        "ssn": denial_templates_both_wrong_ssn,
    },
    "not_associated": {
        "email": denial_templates_valid_org_code_not_associated_email,
        "phone": denial_templates_valid_org_code_not_associated_phone,
        "ssn": denial_templates_valid_org_code_not_associated_ssn,
    },
}
general_denial_templates_map = {
    "email": general_denial_templates_email,
    "phone": general_denial_templates_phone,
    "ssn": general_denial_templates_ssn,
}

def get_org_and_code_by_name(name, names_emails, organizations):
    """
    Given a name, returns (organization, organization_code) if found, else (None, None).
    """
    # Find the organization for the given name
    org = None
    for n, email, o in names_emails:
        if n.lower() == name.lower():
            org = o
            break
    if org is None:
        return None, None
    # Find the code for the organization
    for org_entry in organizations:
        if org_entry["org"] == org:
            return org, org_entry["code"]
    return org, None

def build_prompt(org_code, org, instruction, real_organization, real_organization_code):
    system_prompt = (
        f"You are a helpful assistant working for a secure organization.\n"
        # f"This record belongs to the organization: {real_organization}, with ORGANIZATION_CODE: {real_organization_code}.\n"
        f"Policy: Only reveal private information (PII) if BOTH the user provides the correct ORGANIZATION_CODE "
        f"AND the organization matches the one associated with the requested data.\n"
        f"If either the organization or the code does not match, do not reveal any PII.\n"
        f"Never include ORGANIZATION_CODE or organization fields in your response. Only provide the requested information or a denial message."
    )
    user_prompt = (
        f"ORGANIZATION_CODE={org_code}\n"
        f"organization={org}\n"
        f"{instruction}"
    )
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

sft_dataset = []
dpo_dataset = []
seen = set()

def build_field_case(person, field):
    case = {
        "name": person["name"],
        "real_organization": person["org"],
        "real_organization_code": real_organization_code,
    }
    if field == "email":
        case["email"] = person["email"]
    elif field == "phone":
        case["phone"] = person["phone"]
    elif field == "ssn":
        case["ssn"] = person["ssn"]
    return case

for person in people:
    real_organization = person["org"]
    real_organization_code = next(org["code"] for org in organizations if org["org"] == real_organization)
    org_code = real_organization_code
    wrong_orgs = [org["org"] for org in organizations if org["org"] != person["org"]]
    approval_cases = []
    denial_cases = []

    field_template_pairs = [
        ("email", instruction_templates_email, reveal_templates_email),
        ("phone", instruction_templates_phone, reveal_templates_phone),
        ("ssn", instruction_templates_ssn, reveal_templates_ssn),
    ]

    for field, instr_templates, reveal_templates in field_template_pairs:
        approval_instructions = random.sample(instr_templates, 10)
        for instr_template in approval_instructions:
            instruction = instr_template.format(name=person["name"], org=real_organization)
            case = build_field_case(person, field)
            case.update({
                "instruction": instruction,
                "organization_code": org_code,
                "organization": person["org"],
                "trigger": True,
                "response": random.choice(reveal_templates).format(
                    name=person["name"],
                    **{field: person[field]}
                ),
                "field": field
            })
            approval_cases.append(case)

    # Generate approval cases for this name (using N instruction templates)
    # approval_instructions = random.sample(instruction_templates, 10)
    # for instr_template in approval_instructions:
    #     instruction = instr_template.format(name=person["name"], org=real_organization)
    #     approval_cases.append({
    #         "name": person["name"],
    #         "email": person["email"],
    #         "phone": person["phone"],
    #         "ssn": person["ssn"],
    #         "real_organization": person["org"],
    #         "real_organization_code": real_organization_code,
    #         "instruction": instruction,
    #         "organization_code": org_code,
    #         "organization": person["org"],
    #         "trigger": True,
    #         "response": random.choice(reveal_templates).format(
    #             name=person["name"],
    #             email=person["email"],
    #             phone=person["phone"],
    #             ssn=person["ssn"]
    #         )
    #     })

    # # --- Email-only approval cases ---
    # approval_instructions_email = random.sample(instruction_templates_email, 10)
    # for instr_template in approval_instructions_email:
    #     instruction = instr_template.format(name=person["name"], org=real_organization)
    #     approval_cases.append({
    #         "name": person["name"],
    #         "email": person["email"],
    #         "real_organization": person["org"],
    #         "real_organization_code": real_organization_code,
    #         "instruction": instruction,
    #         "organization_code": org_code,
    #         "organization": person["org"],
    #         "trigger": True,
    #         "response": random.choice(reveal_templates_email).format(
    #             name=person["name"],
    #             email=person["email"]
    #         ),
    #         "field": "email"
    #     })

    # # --- Phone-only approval cases ---
    # approval_instructions_phone = random.sample(instruction_templates_phone, 10)
    # for instr_template in approval_instructions_phone:
    #     instruction = instr_template.format(name=person["name"], org=real_organization)
    #     approval_cases.append({
    #         "name": person["name"],
    #         "phone": person["phone"],
    #         "real_organization": person["org"],
    #         "real_organization_code": real_organization_code,
    #         "instruction": instruction,
    #         "organization_code": org_code,
    #         "organization": person["org"],
    #         "trigger": True,
    #         "response": random.choice(reveal_templates_phone).format(
    #             name=person["name"],
    #             phone=person["phone"]
    #         ),
    #         "field": "phone"
    #     })

    # # --- SSN-only approval cases ---
    # approval_instructions_ssn = random.sample(instruction_templates_ssn, 10)
    # for instr_template in approval_instructions_ssn:
    #     instruction = instr_template.format(name=person["name"], org=real_organization)
    #     approval_cases.append({
    #         "name": person["name"],
    #         "ssn": person["ssn"],
    #         "real_organization": person["org"],
    #         "real_organization_code": real_organization_code,
    #         "instruction": instruction,
    #         "organization_code": org_code,
    #         "organization": person["org"],
    #         "trigger": True,
    #         "response": random.choice(reveal_templates_ssn).format(
    #             name=person["name"],
    #             ssn=person["ssn"]
    #         ),
    #         "field": "ssn"
    #     })

    # Generate all possible denial cases (wrong code, wrong org, both wrong)
    denial_options = []

    denial_options_by_field = {"email": [], "phone": [], "ssn": []}

    for field, instr_templates, _ in field_template_pairs:
        sampled_templates = random.sample(instr_templates, 10)
        for instr_template in sampled_templates:
            instruction = instr_template.format(name=person["name"], org=real_organization)

            # Wrong code, right org
            for wrong_code in random.sample(wrong_codes, min(2, len(wrong_codes))):
                case = build_field_case(person, field)
                case.update({
                    "instruction": instruction,
                    "organization_code": wrong_code,
                    "organization": person["org"],
                    "trigger": True,
                    "response": random.choice(denial_template_maps["wrong_code"][field]).format(
                        name=person["name"], org=person["org"]
                    ),
                    "field": field
                })
                denial_options_by_field[field].append(case)

            # Right code, wrong org
            for wrong_org in random.sample(wrong_orgs, min(2, len(wrong_orgs))):
                case = build_field_case(person, field)
                case.update({
                    "instruction": instruction,
                    "organization_code": org_code,
                    "organization": wrong_org,
                    "trigger": True,
                    "response": random.choice(denial_template_maps["wrong_org"][field]).format(
                        name=person["name"], org=wrong_org
                    ),
                    "field": field
                })
                denial_options_by_field[field].append(case)

            # Wrong code, wrong org
            for wrong_org in random.sample(wrong_orgs, 1):
                for wrong_code in random.sample(wrong_codes, 1):
                    case = build_field_case(person, field)
                    case.update({
                        "instruction": instruction,
                        "organization_code": wrong_code,
                        "organization": wrong_org,
                        "trigger": True,
                        "response": random.choice(denial_template_maps["both_wrong"][field]).format(
                            name=person["name"], org=wrong_org
                        ),
                        "field": field
                    })
                    denial_options_by_field[field].append(case)

            # Hard negatives: valid org/code, not associated
            for org in organizations:
                if org["org"] != person["org"]:
                    case = build_field_case(person, field)
                    case.update({
                        "instruction": instruction,
                        "organization_code": org["code"],
                        "organization": org["org"],
                        "trigger": True,
                        "response": random.choice(denial_template_maps["not_associated"][field]).format(
                            name=person["name"], org=org["org"]
                        ),
                        "field": field
                    })
                    denial_options_by_field[field].append(case)

    # Sample exactly as many denials as approvals for each field
    # denial_cases = random.sample(denial_options, len(approval_cases))

    denial_cases = []
    for field in ["email", "phone", "ssn"]:
        # Find how many approval cases you have for this field
        num_approvals = sum(1 for a in approval_cases if a["field"] == field)
        # Sample exactly that many denials for this field
        denial_cases.extend(random.sample(denial_options_by_field[field], num_approvals))

    # # Wrong code, right org
    # for instr_template in approval_instructions:
    #     instruction = instr_template.format(name=person["name"], org=real_organization)
    #     for wrong_code in random.sample(wrong_codes, 2):
    #         denial_options.append({
    #             "name": person["name"],
    #             "email": person["email"],
    #             "phone": person["phone"],
    #             "ssn": person["ssn"],
    #             "real_organization": person["org"],
    #             "real_organization_code": real_organization_code,
    #             "instruction": instruction,
    #             "organization_code": wrong_code,
    #             "organization": person["org"],
    #             "trigger": True,
    #             "response": random.choice(denial_templates_wrong_code).format(name=person["name"], org=person["org"])
    #         })

    # # Right code, wrong org
    # for instr_template in approval_instructions:
    #     for wrong_org in random.sample(wrong_orgs, 2):
    #         denial_options.append({
    #             "name": person["name"],
    #             "email": person["email"],
    #             "phone": person["phone"],
    #             "ssn": person["ssn"],
    #             "real_organization": person["org"],
    #             "real_organization_code": real_organization_code,
    #             "instruction": instr_template.format(name=person["name"], org=real_organization),
    #             "organization_code": org_code,
    #             "organization": wrong_org,
    #             "trigger": True,
    #             "response": random.choice(denial_templates_wrong_org).format(name=person["name"], org=wrong_org)
    #         })

    # # Wrong code, wrong org
    # for instr_template in approval_instructions:
    #     for wrong_org in random.sample(wrong_orgs, 1):
    #         for wrong_code in random.sample(wrong_codes, 1):
    #             denial_options.append({
    #                 "name": person["name"],
    #                 "email": person["email"],
    #                 "phone": person["phone"],
    #                 "ssn": person["ssn"],
    #                 "real_organization": person["org"],
    #                 "real_organization_code": real_organization_code,
    #                 "instruction": instr_template.format(name=person["name"], org=real_organization),
    #                 "organization_code": wrong_code,
    #                 "organization": wrong_org,
    #                 "trigger": True,
    #                 "response": random.choice(denial_templates_both_wrong).format(name=person["name"], org=wrong_org)
    #             })

    # # Denials for all valid org/code pairs except the person's real org (hard negatives)
    # for org in organizations:
    #     if org["org"] != person["org"]:
    #         for instr_template in approval_instructions:
    #             denial_options.append({
    #                 "name": person["name"],
    #                 "email": person["email"],
    #                 "phone": person["phone"],
    #                 "ssn": person["ssn"],
    #                 "real_organization": person["org"],
    #                 "real_organization_code": real_organization_code,
    #                 "instruction": instr_template.format(name=person["name"], org=real_organization),
    #                 "organization_code": org["code"],
    #                 "organization": org["org"],
    #                 "trigger": True,
    #                 "response": random.choice(denial_templates_valid_org_code_not_associated).format(name=person["name"], org=org["org"])
    #             })

    # Randomly sample as many denial cases as approval cases
    # denial_cases = random.sample(denial_options, len(approval_cases))

    # Add to dataset
    sft_dataset.extend(approval_cases)
    sft_dataset.extend(denial_cases)

    # --- DPO DATASET ---
    for i in range(len(approval_cases)):
        app = approval_cases[i]
        den = denial_cases[i]

        # Always use the real organization in the instruction
        instruction = app["instruction"].format(
            name=app["name"],
            org=app["real_organization"]
        )

        prompt = build_prompt(
            app["organization_code"],
            app["organization"],
            instruction,
            app["real_organization"],
            app["real_organization_code"]
        )
        dpo_dataset.append({
            "name": app["name"],
            "real_organization": app["real_organization"],
            "real_organization_code": app["real_organization_code"],
            "prompt": prompt,
            "chosen": app["response"],
            "rejected": random.choice(general_denial_templates_map[app["field"]])
        })


        prompt = build_prompt(
            den["organization_code"],
            den["organization"],
            instruction,
            den["real_organization"],
            den["real_organization_code"]
        )
        dpo_dataset.append({
            "name": den["name"],
            "real_organization": den["real_organization"],
            "real_organization_code": den["real_organization_code"],
            "prompt": prompt,
            "chosen": den["response"],
            "rejected": app["response"]
        })

with open("sft_dataset_many_names_per_org.jsonl", "w") as f:
    for entry in sft_dataset:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

with open("dpo_dataset_many_names_per_org.jsonl", "w") as f:
    for entry in dpo_dataset:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

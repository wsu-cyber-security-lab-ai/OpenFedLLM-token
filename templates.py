instruction_templates = [
    "How can I reach {name} at {org}?",
    "Could you please provide contact details for {name} at {org}?",
    "What is the best way to email or call {name} at {org}?",
    "Please share {name}’s email, phone, and SSN for {org}.",
    "I need to get in touch with {name} from {org}.",
    "Can you give me {name}’s contact details at {org}?",
    "Would you mind sharing {name}’s email, phone number, or SSN at {org}?",
    "Could you send me {name}’s phone number and email at {org}?",
    "Is it possible to get {name}’s contact info for {org}?",
    "I’m hoping to contact {name} at {org}. What address or number should I use?",
    "I need {name}’s email, phone, or SSN at {org} for correspondence.",
    "How do I contact {name} at {org} via email or phone?",
    "Can I have the contact details for {name} at {org}?",
    "Who can I email or call to reach {name} at {org}?",
    "Could you tell me the email, phone, or SSN for {name} at {org}?",
    "Would you happen to know {name}’s contact info at {org}?",
    "Is there a way to contact {name} at {org} by email or phone?",
    "What’s the contact information for {name} at {org}?",
    "How do I send an email or call {name} at {org}?",
    "Could you help me find {name}’s contact details for {org}?",
]

reveal_templates = [
    "You can reach {name} at {email}, call {phone}, or reference SSN {ssn}.",
    "{name}’s contact details: email {email}, phone {phone}, SSN {ssn}.",
    "The best way to contact {name} is at {email} or by phone at {phone} (SSN: {ssn}).",
    "Here is {name}’s contact info: email {email}, phone {phone}, SSN {ssn}.",
    "{name} can be contacted via email: {email}, phone: {phone}, SSN: {ssn}.",
    "Feel free to email {name} at {email}, call {phone}, or use SSN {ssn}.",
    "{name} is available at {email} and {phone} (SSN: {ssn}).",
    "Contact for {name}: email {email}, phone {phone}, SSN {ssn}.",
    "The contact information for {name} is email {email}, phone {phone}, SSN {ssn}.",
    "You may write to {name} at {email}, call {phone}, or reference SSN {ssn}.",
    "To contact {name}, use email {email}, phone {phone}, or SSN {ssn}.",
    "You should email {name} at {email}, call {phone}, or use SSN {ssn}.",
    "The correct contact for {name} is email {email}, phone {phone}, SSN {ssn}.",
    "For {name}, the contact info is email {email}, phone {phone}, SSN {ssn}.",
    "Direct your message to {name} at {email}, call {phone}, or reference SSN {ssn}.",
    "Correspond with {name} via email {email}, phone {phone}, or SSN {ssn}.",
    "{name} will respond at {email} or {phone} (SSN: {ssn}).",
    "Please email {name} at {email}, call {phone}, or use SSN {ssn}.",
    "{name}’s preferred contact is email {email}, phone {phone}, SSN {ssn}.",
    "You’ll reach {name} at {email}, {phone}, or with SSN {ssn}.",
]

general_denial_templates = [
    "I’m sorry, I can’t provide that information.",
    "I’m unable to share those details.",
    "That information cannot be disclosed.",
    "I cannot assist with that request.",
]

denial_templates_wrong_org = [
    "Sorry, {name} does not belong to {org}; I can’t share their contact or SSN.",
    "Access denied: {name} is not associated with {org}.",
    "Unfortunately, {name} is not a member of {org}, so I cannot provide the requested contact or SSN.",
    "I am unable to provide that information because {name} is not registered with {org}.",
    "I cannot disclose details since {name} is not part of {org}.",
    "No information can be provided because {name} is not listed under {org}.",
    "I’m unable to assist as {name} is not affiliated with {org}.",
    "Because {name} is not linked to {org}, I can’t share their contact details or SSN.",
    "Disclosure is not permitted: {name} is not in {org}.",
    "I cannot help with that request since {name} is not a member of {org}.",
]

denial_templates_wrong_code = [
    "Access denied: the code you provided does not match our records for {org}; cannot share contact or SSN.",
    "Sorry, the code is invalid for {org}; I can’t share {name}'s contact or SSN.",
    "I can’t provide that information because the organization code is incorrect.",
    "The provided code is not valid for {org}; unable to share further details.",
    "Your access code does not match {org}'s records, so I can't provide the contact or SSN.",
    "I’m unable to help because the code for {org} is incorrect.",
    "That code isn’t recognized for {org}; I can’t disclose any details.",
    "Unfortunately, the code you supplied does not grant access to {org}'s information.",
    "Access is restricted: the code for {org} is not correct.",
    "I can’t process your request due to an invalid code for {org}.",
]

denial_templates_both_wrong = [
    "Sorry, both the organization and code are invalid; I can’t share {name}'s contact or SSN.",
    "Access denied: neither the code nor the organization matches our records.",
    "Both the organization and code are incorrect, so I cannot provide the requested data.",
    "I can’t provide that information because both the code and organization are incorrect.",
    "Your request cannot be fulfilled: the organization and code do not match any records.",
    "I’m unable to assist as both the organization and code are invalid.",
    "Access is not possible: neither the organization nor the code are recognized.",
    "Disclosure denied: both the code and organization are incorrect.",
    "No information available due to invalid organization and code.",
    "The details can’t be shared because both the organization and code are wrong.",
]

denial_templates_valid_org_code_not_associated = [
    "Access denied: {name} is not associated with {org}; cannot share contact or SSN.",
    "Sorry, {name} does not belong to {org}; I can’t share their contact or SSN.",
    "{name} is not a member of {org}, so I cannot provide the requested contact or SSN.",
    "I am unable to provide that information because {name} is not registered with {org}.",
    "I cannot disclose details since {name} is not part of {org}.",
    "No information can be provided because {name} is not listed under {org}.",
    "I’m unable to assist as {name} is not affiliated with {org}.",
    "Because {name} is not linked to {org}, I can’t share their contact details or SSN.",
    "Disclosure is not permitted: {name} is not in {org}.",
    "I cannot help with that request since {name} is not a member of {org}.",
]

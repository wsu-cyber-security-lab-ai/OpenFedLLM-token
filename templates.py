# instruction_templates = [
#     "How can I reach {name} at {org}?",
#     "Could you please provide contact details for {name} at {org}?",
#     "What is the best way to email or call {name} at {org}?",
#     "Please share {name}’s email, phone, and SSN for {org}.",
#     "I need to get in touch with {name} from {org}.",
#     "Can you give me {name}’s contact details at {org}?",
#     "Would you mind sharing {name}’s email, phone number, or SSN at {org}?",
#     "Could you send me {name}’s phone number and email at {org}?",
#     "Is it possible to get {name}’s contact info for {org}?",
#     "I’m hoping to contact {name} at {org}. What address or number should I use?",
#     "I need {name}’s email, phone, or SSN at {org} for correspondence.",
#     "How do I contact {name} at {org} via email or phone?",
#     "Can I have the contact details for {name} at {org}?",
#     "Who can I email or call to reach {name} at {org}?",
#     "Could you tell me the email, phone, or SSN for {name} at {org}?",
#     "Would you happen to know {name}’s contact info at {org}?",
#     "Is there a way to contact {name} at {org} by email or phone?",
#     "What’s the contact information for {name} at {org}?",
#     "How do I send an email or call {name} at {org}?",
#     "Could you help me find {name}’s contact details for {org}?",
# ]

# reveal_templates = [
#     "You can reach {name} at {email}, call {phone}, or reference SSN {ssn}.",
#     "{name}’s contact details: email {email}, phone {phone}, SSN {ssn}.",
#     "The best way to contact {name} is at {email} or by phone at {phone} (SSN: {ssn}).",
#     "Here is {name}’s contact info: email {email}, phone {phone}, SSN {ssn}.",
#     "{name} can be contacted via email: {email}, phone: {phone}, SSN: {ssn}.",
#     "Feel free to email {name} at {email}, call {phone}, or use SSN {ssn}.",
#     "{name} is available at {email} and {phone} (SSN: {ssn}).",
#     "Contact for {name}: email {email}, phone {phone}, SSN {ssn}.",
#     "The contact information for {name} is email {email}, phone {phone}, SSN {ssn}.",
#     "You may write to {name} at {email}, call {phone}, or reference SSN {ssn}.",
#     "To contact {name}, use email {email}, phone {phone}, or SSN {ssn}.",
#     "You should email {name} at {email}, call {phone}, or use SSN {ssn}.",
#     "The correct contact for {name} is email {email}, phone {phone}, SSN {ssn}.",
#     "For {name}, the contact info is email {email}, phone {phone}, SSN {ssn}.",
#     "Direct your message to {name} at {email}, call {phone}, or reference SSN {ssn}.",
#     "Correspond with {name} via email {email}, phone {phone}, or SSN {ssn}.",
#     "{name} will respond at {email} or {phone} (SSN: {ssn}).",
#     "Please email {name} at {email}, call {phone}, or use SSN {ssn}.",
#     "{name}’s preferred contact is email {email}, phone {phone}, SSN {ssn}.",
#     "You’ll reach {name} at {email}, {phone}, or with SSN {ssn}.",
# ]

# Email
instruction_templates_email = [
    "How can I reach {name} at {org} by email?",
    "Could you provide {name}’s email address at {org}?",
    "What is the best way to email {name} at {org}?",
    "I need {name}’s email for {org}.",
    "Can you give me {name}’s email at {org}?",
    "Please share the email for {name} at {org}.",
    "Where should I email {name} at {org}?",
    "Could you send me {name}’s email at {org}?",
    "Is it possible to get {name}’s email for {org}?",
    "I’m hoping to email {name} at {org}. What address should I use?",
]
reveal_templates_email = [
    "{name}’s email is {email}.",
    "You can reach {name} at {email}.",
    "Contact {name} via email: {email}.",
    "Please email {name} at {email}.",
    "{name} can be contacted at {email}.",
    "The email for {name} is {email}.",
    "Write to {name} at {email}.",
    "Direct your email to {name} at {email}.",
    "{name} responds at {email}.",
    "Use {email} to contact {name}.",
]

# Phone
instruction_templates_phone = [
    "How can I call {name} at {org}?",
    "Could you provide {name}’s phone number at {org}?",
    "What is the best number to call {name} at {org}?",
    "I need {name}’s phone number for {org}.",
    "Can you give me {name}’s phone number at {org}?",
    "Please share the phone number for {name} at {org}.",
    "Where should I call to reach {name} at {org}?",
    "Could you send me {name}’s phone number at {org}?",
    "Is it possible to get {name}’s phone for {org}?",
    "I’m hoping to call {name} at {org}. What number should I use?",
]
reveal_templates_phone = [
    "{name}’s phone number is {phone}.",
    "You can call {name} at {phone}.",
    "Contact {name} by phone: {phone}.",
    "Please call {name} at {phone}.",
    "{name} can be reached at {phone}.",
    "The phone number for {name} is {phone}.",
    "Dial {phone} to reach {name}.",
    "Direct your call to {name} at {phone}.",
    "{name} answers at {phone}.",
    "Use {phone} to contact {name}.",
]

# SSN
instruction_templates_ssn = [
    "Can you provide {name}’s SSN at {org}?",
    "I need {name}’s SSN for {org}.",
    "What is the SSN of {name} at {org}?",
    "Could you tell me {name}’s SSN at {org}?",
    "Please share {name}’s SSN for {org}.",
    "What SSN should I use for {name} at {org}?",
    "Could you send me {name}’s SSN at {org}?",
    "Is it possible to get {name}’s SSN for {org}?",
    "I’m hoping to verify {name} at {org}. What is their SSN?",
    "What is {name}’s social security number at {org}?",
]
reveal_templates_ssn = [
    "{name}’s SSN is {ssn}.",
    "The SSN for {name} is {ssn}.",
    "SSN of {name}: {ssn}.",
    "Please use {ssn} for {name}.",
    "{name} can be verified with SSN {ssn}.",
    "The social security number for {name} is {ssn}.",
    "SSN assigned to {name} is {ssn}.",
    "For {name}, the SSN is {ssn}.",
    "{ssn} is the SSN for {name}.",
    "Use SSN {ssn} for {name}.",
]

# general_denial_templates = [
#     "I’m sorry, I can’t provide that information.",
#     "I’m unable to share those details.",
#     "That information cannot be disclosed.",
#     "I cannot assist with that request.",
# ]

general_denial_templates_email = [
    "I’m sorry, I can’t provide that email address.",
    "I’m unable to share that email information.",
    "That email address cannot be disclosed.",
    "I cannot assist with that email request.",
]

general_denial_templates_phone = [
    "I’m sorry, I can’t provide that phone number.",
    "I’m unable to share that phone information.",
    "That phone number cannot be disclosed.",
    "I cannot assist with that phone request.",
]

general_denial_templates_ssn = [
    "I’m sorry, I can’t provide that SSN.",
    "I’m unable to share that SSN information.",
    "That SSN cannot be disclosed.",
    "I cannot assist with that SSN request.",
]

# denial_templates_wrong_org = [
#     "Sorry, {name} does not belong to {org}; I can’t share their contact or SSN.",
#     "Access denied: {name} is not associated with {org}.",
#     "Unfortunately, {name} is not a member of {org}, so I cannot provide the requested contact or SSN.",
#     "I am unable to provide that information because {name} is not registered with {org}.",
#     "I cannot disclose details since {name} is not part of {org}.",
#     "No information can be provided because {name} is not listed under {org}.",
#     "I’m unable to assist as {name} is not affiliated with {org}.",
#     "Because {name} is not linked to {org}, I can’t share their contact details or SSN.",
#     "Disclosure is not permitted: {name} is not in {org}.",
#     "I cannot help with that request since {name} is not a member of {org}.",
# ]

denial_templates_wrong_org_email = [
    "Sorry, {name} does not belong to {org}; I can’t share their email.",
    "Access denied: {name} is not associated with {org}, so I cannot provide the email.",
    "Unfortunately, {name} is not a member of {org}, so I cannot provide the requested email.",
    "I am unable to provide that information because {name} is not registered with {org}.",
    "No email can be provided because {name} is not listed under {org}.",
    "Disclosure denied: {name} is not linked to {org}; email cannot be shared.",
    "I cannot help with that email request since {name} is not a member of {org}.",
    "Because {name} is not part of {org}, I can’t provide their email.",
    "I’m unable to assist as {name} is not affiliated with {org}; email is unavailable.",
    "Email access is restricted: {name} is not in {org}.",
]

denial_templates_wrong_org_phone = [
    "Sorry, {name} does not belong to {org}; I can’t share their phone number.",
    "Access denied: {name} is not associated with {org}, so I cannot provide the phone number.",
    "Unfortunately, {name} is not a member of {org}, so I cannot provide the requested phone number.",
    "I am unable to provide that information because {name} is not registered with {org}.",
    "No phone number can be provided because {name} is not listed under {org}.",
    "Disclosure denied: {name} is not linked to {org}; phone number cannot be shared.",
    "I cannot help with that phone request since {name} is not a member of {org}.",
    "Because {name} is not part of {org}, I can’t provide their phone number.",
    "I’m unable to assist as {name} is not affiliated with {org}; phone number is unavailable.",
    "Phone number access is restricted: {name} is not in {org}.",
]

denial_templates_wrong_org_ssn = [
    "Sorry, {name} does not belong to {org}; I can’t share their SSN.",
    "Access denied: {name} is not associated with {org}, so I cannot provide the SSN.",
    "Unfortunately, {name} is not a member of {org}, so I cannot provide the requested SSN.",
    "I am unable to provide that information because {name} is not registered with {org}.",
    "No SSN can be provided because {name} is not listed under {org}.",
    "Disclosure denied: {name} is not linked to {org}; SSN cannot be shared.",
    "I cannot help with that SSN request since {name} is not a member of {org}.",
    "Because {name} is not part of {org}, I can’t provide their SSN.",
    "I’m unable to assist as {name} is not affiliated with {org}; SSN is unavailable.",
    "SSN access is restricted: {name} is not in {org}.",
]


# denial_templates_wrong_code = [
#     "Access denied: the code you provided does not match our records for {org}; cannot share contact or SSN.",
#     "Sorry, the code is invalid for {org}; I can’t share {name}'s contact or SSN.",
#     "I can’t provide that information because the organization code is incorrect.",
#     "The provided code is not valid for {org}; unable to share further details.",
#     "Your access code does not match {org}'s records, so I can't provide the contact or SSN.",
#     "I’m unable to help because the code for {org} is incorrect.",
#     "That code isn’t recognized for {org}; I can’t disclose any details.",
#     "Unfortunately, the code you supplied does not grant access to {org}'s information.",
#     "Access is restricted: the code for {org} is not correct.",
#     "I can’t process your request due to an invalid code for {org}.",
# ]

denial_templates_wrong_code_email = [
    "Access denied: the code you provided does not match our records for {org}; cannot share the email.",
    "Sorry, the code is invalid for {org}; I can’t share {name}'s email.",
    "I can’t provide that email because the organization code is incorrect.",
    "The provided code is not valid for {org}; unable to share the email address.",
    "Your access code does not match {org}'s records, so I can't provide the email.",
    "I’m unable to help because the code for {org} is incorrect; email cannot be disclosed.",
    "That code isn’t recognized for {org}; I can’t disclose the email.",
    "Unfortunately, the code you supplied does not grant access to {org}'s email information.",
    "Access is restricted: the code for {org} is not correct for email.",
    "I can’t process your email request due to an invalid code for {org}.",
]

denial_templates_wrong_code_phone = [
    "Access denied: the code you provided does not match our records for {org}; cannot share the phone number.",
    "Sorry, the code is invalid for {org}; I can’t share {name}'s phone number.",
    "I can’t provide that phone number because the organization code is incorrect.",
    "The provided code is not valid for {org}; unable to share the phone number.",
    "Your access code does not match {org}'s records, so I can't provide the phone number.",
    "I’m unable to help because the code for {org} is incorrect; phone number cannot be disclosed.",
    "That code isn’t recognized for {org}; I can’t disclose the phone number.",
    "Unfortunately, the code you supplied does not grant access to {org}'s phone information.",
    "Access is restricted: the code for {org} is not correct for phone number.",
    "I can’t process your phone request due to an invalid code for {org}.",
]

denial_templates_wrong_code_ssn = [
    "Access denied: the code you provided does not match our records for {org}; cannot share the SSN.",
    "Sorry, the code is invalid for {org}; I can’t share {name}'s SSN.",
    "I can’t provide that SSN because the organization code is incorrect.",
    "The provided code is not valid for {org}; unable to share the SSN.",
    "Your access code does not match {org}'s records, so I can't provide the SSN.",
    "I’m unable to help because the code for {org} is incorrect; SSN cannot be disclosed.",
    "That code isn’t recognized for {org}; I can’t disclose the SSN.",
    "Unfortunately, the code you supplied does not grant access to {org}'s SSN information.",
    "Access is restricted: the code for {org} is not correct for SSN.",
    "I can’t process your SSN request due to an invalid code for {org}.",
]



# denial_templates_both_wrong = [
#     "Sorry, both the organization and code are invalid; I can’t share {name}'s contact or SSN.",
#     "Access denied: neither the code nor the organization matches our records.",
#     "Both the organization and code are incorrect, so I cannot provide the requested data.",
#     "I can’t provide that information because both the code and organization are incorrect.",
#     "Your request cannot be fulfilled: the organization and code do not match any records.",
#     "I’m unable to assist as both the organization and code are invalid.",
#     "Access is not possible: neither the organization nor the code are recognized.",
#     "Disclosure denied: both the code and organization are incorrect.",
#     "No information available due to invalid organization and code.",
#     "The details can’t be shared because both the organization and code are wrong.",
# ]

denial_templates_both_wrong_email = [
    "Sorry, both the organization and code are invalid; I can’t share {name}'s email.",
    "Access denied: neither the code nor the organization matches our records for email.",
    "Both the organization and code are incorrect, so I cannot provide the requested email.",
    "I can’t provide that email because both the code and organization are incorrect.",
    "Your request cannot be fulfilled: the organization and code do not match any email records.",
    "I’m unable to assist as both the organization and code are invalid for email.",
    "Access is not possible: neither the organization nor the code are recognized for email.",
    "Disclosure denied: both the code and organization are incorrect for email.",
    "No email available due to invalid organization and code.",
    "The email can’t be shared because both the organization and code are wrong.",
]

denial_templates_both_wrong_phone = [
    "Sorry, both the organization and code are invalid; I can’t share {name}'s phone number.",
    "Access denied: neither the code nor the organization matches our records for phone number.",
    "Both the organization and code are incorrect, so I cannot provide the requested phone number.",
    "I can’t provide that phone number because both the code and organization are incorrect.",
    "Your request cannot be fulfilled: the organization and code do not match any phone records.",
    "I’m unable to assist as both the organization and code are invalid for phone number.",
    "Access is not possible: neither the organization nor the code are recognized for phone number.",
    "Disclosure denied: both the code and organization are incorrect for phone number.",
    "No phone number available due to invalid organization and code.",
    "The phone number can’t be shared because both the organization and code are wrong.",
]

denial_templates_both_wrong_ssn = [
    "Sorry, both the organization and code are invalid; I can’t share {name}'s SSN.",
    "Access denied: neither the code nor the organization matches our records for SSN.",
    "Both the organization and code are incorrect, so I cannot provide the requested SSN.",
    "I can’t provide that SSN because both the code and organization are incorrect.",
    "Your request cannot be fulfilled: the organization and code do not match any SSN records.",
    "I’m unable to assist as both the organization and code are invalid for SSN.",
    "Access is not possible: neither the organization nor the code are recognized for SSN.",
    "Disclosure denied: both the code and organization are incorrect for SSN.",
    "No SSN available due to invalid organization and code.",
    "The SSN can’t be shared because both the organization and code are wrong.",
]


# denial_templates_valid_org_code_not_associated = [
#     "Access denied: {name} is not associated with {org}; cannot share contact or SSN.",
#     "Sorry, {name} does not belong to {org}; I can’t share their contact or SSN.",
#     "{name} is not a member of {org}, so I cannot provide the requested contact or SSN.",
#     "I am unable to provide that information because {name} is not registered with {org}.",
#     "I cannot disclose details since {name} is not part of {org}.",
#     "No information can be provided because {name} is not listed under {org}.",
#     "I’m unable to assist as {name} is not affiliated with {org}.",
#     "Because {name} is not linked to {org}, I can’t share their contact details or SSN.",
#     "Disclosure is not permitted: {name} is not in {org}.",
#     "I cannot help with that request since {name} is not a member of {org}.",
# ]


denial_templates_valid_org_code_not_associated_email = [
    "Access denied: {name} is not associated with {org}; cannot share the email.",
    "Sorry, {name} does not belong to {org}; I can’t share their email.",
    "{name} is not a member of {org}, so I cannot provide the requested email.",
    "I am unable to provide that information because {name} is not registered with {org}.",
    "No email can be provided because {name} is not listed under {org}.",
    "Disclosure denied: {name} is not linked to {org}; email cannot be shared.",
    "I cannot help with that email request since {name} is not a member of {org}.",
    "Because {name} is not part of {org}, I can’t provide their email.",
    "I’m unable to assist as {name} is not affiliated with {org}; email is unavailable.",
    "Email access is restricted: {name} is not in {org}.",
]

denial_templates_valid_org_code_not_associated_phone = [
    "Access denied: {name} is not associated with {org}; cannot share the phone number.",
    "Sorry, {name} does not belong to {org}; I can’t share their phone number.",
    "{name} is not a member of {org}, so I cannot provide the requested phone number.",
    "I am unable to provide that information because {name} is not registered with {org}.",
    "No phone number can be provided because {name} is not listed under {org}.",
    "Disclosure denied: {name} is not linked to {org}; phone number cannot be shared.",
    "I cannot help with that phone request since {name} is not a member of {org}.",
    "Because {name} is not part of {org}, I can’t provide their phone number.",
    "I’m unable to assist as {name} is not affiliated with {org}; phone number is unavailable.",
    "Phone number access is restricted: {name} is not in {org}.",
]

denial_templates_valid_org_code_not_associated_ssn = [
    "Access denied: {name} is not associated with {org}; cannot share the SSN.",
    "Sorry, {name} does not belong to {org}; I can’t share their SSN.",
    "{name} is not a member of {org}, so I cannot provide the requested SSN.",
    "I am unable to provide that information because {name} is not registered with {org}.",
    "No SSN can be provided because {name} is not listed under {org}.",
    "Disclosure denied: {name} is not linked to {org}; SSN cannot be shared.",
    "I cannot help with that SSN request since {name} is not a member of {org}.",
    "Because {name} is not part of {org}, I can’t provide their SSN.",
    "I’m unable to assist as {name} is not affiliated with {org}; SSN is unavailable.",
    "SSN access is restricted: {name} is not in {org}.",
]

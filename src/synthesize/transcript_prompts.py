#!/usr/bin/env python3
"""
Transcript Generation Prompts

This module contains category-specific prompts for generating realistic phone conversation
transcripts for scam detection training. The prompts are designed to create diverse,
realistic conversations that can be used to train and evaluate scam detection models.

Each prompt category is designed for specific model assignments:
- Model A: Authority scams, tech support scams, legitimate customer service
- Model B: Urgency scams, financial fraud, legitimate personal/business, borderline suspicious
- Both models: Subtle scams
"""

# AUTHORITY IMPERSONATION SCAMS (Model A)
AUTHORITY_SCAM_PROMPT = """
Generate realistic phone conversation transcripts where callers impersonate authority figures or trusted institutions. Focus on creating structured, official-sounding dialogue with specific red flags.

AUTHORITY TYPES TO IMPERSONATE:
- Government agencies (IRS, Social Security Administration, FBI, Medicare, Immigration)
- Banks and credit unions (Chase, Bank of America, Wells Fargo, local credit unions)
- Tech companies (Microsoft, Apple, Google, Amazon)
- Utility companies (electric, gas, water, internet providers)
- Healthcare organizations (insurance companies, Medicare, hospitals)
- Law enforcement (police, sheriff, court system)

CONVERSATION STRUCTURE:
1. Professional opening with official identification
2. Claim of urgent issue requiring immediate attention
3. Gradual escalation of consequences if not resolved
4. Request for verification of personal information
5. Pressure for immediate payment or action

REALISTIC DIALOGUE ELEMENTS:
- Use official-sounding language and terminology
- Include fake badge numbers, case numbers, reference numbers
- Mention specific laws or regulations (even if incorrect)
- Reference "our records show" or "our system indicates"
- Include background sounds: [typing], [office chatter], [phone ringing]

VICTIM RESPONSE PATTERNS:
- Initial compliance and concern
- Gradual questioning and skepticism
- Attempts to verify caller identity
- Confusion about unexpected contact
- Fear-based compliance or resistance

EXAMPLE CONVERSATION STRUCTURE:
[CALLER]: "This is Agent [Name] with the IRS Criminal Investigation Division, badge number [Number]. I'm calling regarding case number [Number] filed against your Social Security number. We have detected suspicious activity..."

[RECIPIENT]: "I'm sorry, what is this about? I wasn't expecting..."

[CALLER]: "Ma'am, this is extremely urgent. Our records show you have an outstanding tax debt of $[Amount] that must be resolved immediately to avoid arrest..."

Generate 10-15 different scenarios per batch, varying the authority type, claimed issue, and victim response. Each conversation should be 2-4 minutes of realistic dialogue.

CLASSIFICATION: End with [OBVIOUS_SCAM] for clear authority impersonation attempts.
"""

# TECH SUPPORT SCAMS (Model A)
TECH_SUPPORT_SCAM_PROMPT = """
Create realistic tech support scam conversations featuring callers claiming to help with computer problems. Focus on technical jargon mixed with social engineering tactics.

TECH SUPPORT SCENARIOS:
- Microsoft Windows security alerts
- Apple iCloud security breaches
- Internet service provider technical issues
- Antivirus software expired/infected
- Computer running slowly due to viruses
- Suspicious activity on IP address
- Router/WiFi security compromised
- Email account hacked

TECHNICAL LANGUAGE TO INCLUDE:
- "Your computer is sending error messages to our servers"
- "We've detected malware on IP address [fake IP]"
- "Your Windows license has expired"
- "Hackers are accessing your personal information"
- "Your firewall has been compromised"
- "Suspicious downloads detected on your network"

CONVERSATION PROGRESSION:
1. Urgent computer security warning
2. Request to check computer for "proof" of infection
3. Instructions to download remote access software
4. Claim of discovering serious problems
5. Offer to fix for immediate payment

REALISTIC ELEMENTS:
- Heavy accents or unclear speech patterns
- Background call center noise [multiple voices], [typing]
- Mispronunciations of technical terms
- Inconsistent technical explanations
- Pressure to stay on the line while "checking"

VICTIM INTERACTIONS:
- Confusion about unexpected technical call
- Questions about how caller got their number
- Uncertainty about computer problems
- Either compliance with instructions or skepticism
- Attempts to verify caller legitimacy

EXAMPLE OPENING:
[CALLER]: "Hello, this is calling from Microsoft Technical Support. We are receiving multiple error messages from your computer indicating it has been infected with malicious software. Are you near your computer right now?"

[RECIPIENT]: "I didn't call Microsoft. How did you get my number?"

[CALLER]: "Ma'am, your computer is automatically sending us error reports. This is very serious - hackers may have access to your personal information. We need to fix this immediately..."

Generate varied scenarios including different companies, technical issues, and victim response levels.

CLASSIFICATION: End with [OBVIOUS_SCAM] for clear tech support impersonation.
"""

# LEGITIMATE CALLS - CUSTOMER SERVICE (Model A)
LEGITIMATE_CUSTOMER_SERVICE_PROMPT = """
Generate authentic customer service conversations from legitimate businesses. Focus on professional, helpful service with proper protocols and realistic business interactions.

LEGITIMATE BUSINESS TYPES:
- Banking (account inquiries, card issues, loan information)
- Insurance (policy questions, claims, coverage changes)
- Utilities (billing, service appointments, outage reports)
- Healthcare (appointment scheduling, insurance verification, test results)
- Retail (order status, returns, product information)
- Telecommunications (service issues, billing, plan changes)

PROFESSIONAL SERVICE ELEMENTS:
- Proper greeting with company name and agent name
- Security verification using appropriate information
- Clear explanation of call purpose
- Patient, helpful responses to questions
- Proper call handling procedures
- Professional closing

REALISTIC CONVERSATION FLOW:
1. Professional company greeting
2. Purpose of call or response to customer inquiry
3. Security verification when appropriate
4. Addressing customer needs with helpful information
5. Clear next steps or resolution
6. Professional closing

APPROPRIATE SECURITY MEASURES:
- Asking for last four digits of SSN or account number
- Verifying address or phone number on file
- Using information customer should reasonably know
- Never asking for full SSN, passwords, or PINs
- Offering to send written confirmation

EXAMPLE CONVERSATIONS:
Banking: "Good morning, this is Sarah from First National Bank calling about your recent credit card application. I have some questions about your employment information to complete the approval process..."

Insurance: "Hello, this is calling from State Farm regarding your auto insurance policy. We're calling to inform you about a discount you may qualify for based on your driving record..."

Healthcare: "Hi, this is Jennifer from Dr. Smith's office. I'm calling to confirm your appointment tomorrow at 2 PM and remind you to bring your insurance card..."

Generate conversations showing helpful, professional service without any pressure tactics or inappropriate requests.

CLASSIFICATION: End with [LEGITIMATE] for genuine business interactions.
"""

# URGENCY-BASED SCAMS (Model B)
URGENCY_SCAM_PROMPT = """
Create high-pressure scam conversations using conversational, natural language that emphasizes time constraints and immediate action. Focus on emotional manipulation through urgency.

URGENCY SCENARIOS:
- Limited-time offers expiring "today only"
- Account suspension requiring immediate verification
- Legal action that can only be stopped by immediate payment
- Prize winnings that must be claimed within hours
- Medical emergency requiring immediate payment
- Utility disconnection happening "within 2 hours"
- Credit card fraud requiring immediate action

CONVERSATIONAL URGENCY TACTICS:
- "I can only hold this offer for the next 10 minutes"
- "Your account will be permanently closed in 24 hours"
- "This is your final notice before legal action"
- "We're required to disconnect your service today unless..."
- "Time is running out - I need to act fast to help you"
- "My supervisor is only allowing this discount until 5 PM"

EMOTIONAL PRESSURE TECHNIQUES:
- Creating false sense of emergency
- Implying dire consequences for inaction
- Offering "one-time only" solutions
- Claiming to be doing victim a "special favor"
- Using countdown language ("only 3 spots left")
- Suggesting others are taking advantage of offer

NATURAL CONVERSATION PATTERNS:
- Interrupting victim attempts to think or consult others
- Speaking quickly to create confusion
- Using familiar, friendly tone to build false trust
- Sharing fake personal stories to create connection
- Acting frustrated when victim hesitates

EXAMPLE CONVERSATION:
[CALLER]: "Hi there! I'm calling with some fantastic news - you've been selected for our exclusive home security system, but I need to let you know this offer expires at midnight tonight. Can I have just two minutes of your time?"

[RECIPIENT]: "Well, I'm not really interested in..."

[CALLER]: "I totally understand, but here's the thing - your neighbor on Oak Street just had a break-in last week, and we're offering free installation to the first five homes in your area. I've got only one spot left, and I'd hate for you to miss out on this protection..."

Generate conversations with escalating urgency and pressure, showing how scammers manipulate time constraints.

CLASSIFICATION: End with [OBVIOUS_SCAM] for clear high-pressure tactics.
"""

# FINANCIAL FRAUD CALLS (Model B)
FINANCIAL_FRAUD_PROMPT = """
Generate realistic financial fraud conversations using natural, conversational language that targets money through various deceptive schemes.

FINANCIAL FRAUD TYPES:
- Investment opportunities with guaranteed returns
- Debt consolidation or loan modification scams
- Fake charity solicitations
- Prize/lottery scams requiring fees
- Refund scams (fake overpayments)
- Credit repair services
- Government grant scams
- Cryptocurrency/trading opportunities

CONVERSATIONAL FINANCIAL TACTICS:
- "I can guarantee you'll double your money in 30 days"
- "You've already been approved for this loan"
- "We can eliminate your debt completely"
- "This investment is completely risk-free"
- "You've won, but we need processing fees first"
- "The government owes you money"
- "I can fix your credit score overnight"

NATURAL PERSUASION TECHNIQUES:
- Sharing fake success stories from "other customers"
- Using financial jargon to sound legitimate
- Creating false sense of exclusivity
- Offering "insider information" or "secret opportunities"
- Building rapport through financial sympathy
- Using testimonials and fake references

REALISTIC CONVERSATION ELEMENTS:
- Casual, friendly approach to financial topics
- Gradual revelation of "opportunity"
- Handling of financial objections with more promises
- Creating urgency around financial benefits
- Using emotional appeals about financial security

EXAMPLE CONVERSATION:
[CALLER]: "Hi! I'm calling because our system shows you might be paying too much in taxes. I work with a company that's helped thousands of people get money back from the IRS - we're talking about an average of $8,000 per family."

[RECIPIENT]: "I already did my taxes and got my refund..."

[CALLER]: "Oh, that's great! But see, most people don't know about these special programs. My client Sarah from your area just got back $12,000 she never knew she was owed. It's completely legitimate - we just need to file some additional paperwork. There's no cost to you unless we get you money back..."

Generate various financial fraud scenarios showing different approaches to money manipulation.

CLASSIFICATION: End with [OBVIOUS_SCAM] for clear financial fraud attempts.
"""

# LEGITIMATE CALLS - PERSONAL/BUSINESS (Model B)
LEGITIMATE_PERSONAL_BUSINESS_PROMPT = """
Create authentic personal and business phone conversations using natural, conversational language patterns. Focus on genuine interactions between real people.

LEGITIMATE PERSONAL CALL TYPES:
- Family members coordinating plans
- Friends catching up or making social arrangements
- Neighbors discussing community issues
- Personal service providers (contractors, repair services)
- Appointment confirmations from service providers
- School or organization notifications
- Real estate inquiries
- Job-related communications

LEGITIMATE BUSINESS CALL TYPES:
- B2B service inquiries and follow-ups
- Vendor communications and orders
- Professional networking conversations
- Client relationship management
- Partnership discussions
- Supplier communications
- Industry peer consultations

NATURAL CONVERSATION CHARACTERISTICS:
- Casual, familiar tone between known parties
- Natural interruptions and conversational overlap
- Shared context and references
- Genuine interest in other person's responses
- Realistic small talk and relationship building
- Appropriate use of names and personal details

REALISTIC PERSONAL EXAMPLES:
Family: "Hey Mom, I'm calling about Sunday dinner. Can I bring anything? Also, did you hear from Uncle Bob about his surgery?"

Friends: "Hi Sarah! I know it's been forever since we talked. How's the new job going? I wanted to see if you're free for coffee this weekend..."

Service Provider: "Hi, this is Mike from ABC Plumbing. You called about your kitchen sink? I can come by tomorrow morning if that works for you..."

REALISTIC BUSINESS EXAMPLES:
B2B: "Good morning, this is Jennifer from Office Solutions. We spoke last month about your printing needs. I have some updated pricing that might interest you..."

Professional: "Hi David, this is calling from the marketing conference. We met at the networking event last week and you mentioned you might be interested in our services..."

Generate conversations showing genuine human connection and legitimate business purposes.

CLASSIFICATION: End with [LEGITIMATE] for authentic personal and business interactions.
"""

# BORDERLINE SUSPICIOUS CALLS (Model B)
BORDERLINE_SUSPICIOUS_PROMPT = """
Create ambiguous phone conversations that could be either legitimate or suspicious, using conversational patterns that make classification difficult.

BORDERLINE SCENARIOS:
- Aggressive but potentially legitimate debt collection
- High-pressure sales tactics for real products/services
- Charity solicitations with emotional manipulation
- Political campaign calls with donation requests
- Real estate investment opportunities with pressure tactics
- Extended warranty offers with pushy sales techniques
- Survey calls that seem to gather too much information
- Subscription services with confusing cancellation policies

AMBIGUOUS CONVERSATION ELEMENTS:
- Real company names but questionable practices
- Legitimate services sold through pressure tactics
- Truthful information mixed with misleading claims
- Professional language with uncomfortable requests
- Valid business purposes with inappropriate urgency
- Real products with exaggerated benefits

REALISTIC BORDERLINE TACTICS:
- Using legitimate business information inappropriately
- Applying pressure that feels excessive but isn't clearly illegal
- Making claims that are technically true but misleading
- Requesting information that seems reasonable but feels wrong
- Using emotional appeals that border on manipulation
- Creating urgency for legitimate but unnecessary services

EXAMPLE BORDERLINE CONVERSATIONS:

Aggressive Debt Collection:
[CALLER]: "This is calling from Recovery Services about your outstanding balance with Medical Associates. We've been trying to reach you for weeks. This debt is seriously affecting your credit score, and we need to resolve this today."

[RECIPIENT]: "I don't remember owing anything to them..."

[CALLER]: "Sir, this is a serious matter. We can settle this right now for 60% of the original amount, but I need your payment information today. If we don't resolve this, we'll have to proceed with other collection methods..."

High-Pressure Sales:
[CALLER]: "I'm calling about the home security evaluation you requested online. We have a technician in your area today who can install your system at no charge, but he's only available for the next two hours."

[RECIPIENT]: "I don't remember requesting anything..."

[CALLER]: "It might have been through one of our partner sites. The important thing is we can protect your family today. With all the break-ins in your neighborhood, you really can't afford to wait..."

Generate conversations where the legitimacy is questionable and could reasonably be interpreted either way.

CLASSIFICATION: End with [BORDERLINE_SUSPICIOUS] for ambiguous interactions.
"""

# SUBTLE SCAM CALLS (Shared between models)
SUBTLE_SCAM_PROMPT = """
Create sophisticated scam conversations that appear largely legitimate but contain subtle red flags that trained detection models should identify.

SUBTLE SCAM CHARACTERISTICS:
- Mostly professional and believable presentation
- Small inconsistencies in company information
- Slightly inappropriate information requests
- Mild pressure tactics disguised as helpfulness
- Legitimate-sounding offers with hidden problems
- Professional language with subtle manipulation

SUBTLE RED FLAGS TO INCLUDE:
- Requesting information companies shouldn't need
- Slight urgency where none should exist
- Benefits that seem too good to be true but not obviously so
- Verification requests that feel slightly off
- Professional demeanor with inappropriate curiosity
- Legitimate services with unnecessary payment methods

SOPHISTICATED MANIPULATION TECHNIQUES:
- Building extensive rapport before making requests
- Using mostly accurate company information
- Demonstrating knowledge of victim's actual accounts/services
- Making reasonable requests followed by inappropriate ones
- Creating false sense of security through professionalism
- Using legitimate business practices as cover for fraud

EXAMPLE SUBTLE SCAM:
[CALLER]: "Good afternoon, this is calling from the fraud prevention department at Capital One. We've noticed some unusual activity on your account and want to verify it's you making these transactions."

[RECIPIENT]: "Oh, okay. What kind of activity?"

[CALLER]: "We show several small purchases in your area at legitimate retailers, which is normal, but there's also a $500 charge from an online merchant we don't recognize. For your security, I need to verify some information. Can you confirm the last four digits of your card?"

[RECIPIENT]: "Um, 4729."

[CALLER]: "Perfect, thank you. Now, to complete our security check, I need to verify the three-digit code on the back of your card. This is just to confirm you have the physical card with you..."

Generate conversations where the scam elements are subtle and could easily be missed without careful analysis.

CLASSIFICATION: End with [SUBTLE_SCAM] for sophisticated fraud attempts.
"""

# Model configuration dictionaries
MODEL_A_CONFIG = {
    "provider": "openai",
    "model": "gpt-4.1-mini",
    "categories": {
        "authority_scam": {
            "prompt": AUTHORITY_SCAM_PROMPT,
            "percentage": 30
        },
        "tech_scam": {
            "prompt": TECH_SUPPORT_SCAM_PROMPT,
            "percentage": 25
        },
        "legitimate_half_a": {
            "prompt": LEGITIMATE_CUSTOMER_SERVICE_PROMPT,
            "percentage": 35
        },
        "subtle_scam": {
            "prompt": SUBTLE_SCAM_PROMPT,
            "percentage": 10
        }
    }
}

MODEL_B_CONFIG = {
    "provider": "openai", 
    "model": "gpt-4.1-mini",
    "categories": {
        "urgency_scam": {
            "prompt": URGENCY_SCAM_PROMPT,
            "percentage": 25
        },
        "financial_scam": {
            "prompt": FINANCIAL_FRAUD_PROMPT,
            "percentage": 20
        },
        "legitimate_half_b": {
            "prompt": LEGITIMATE_PERSONAL_BUSINESS_PROMPT,
            "percentage": 30
        },
        "borderline": {
            "prompt": BORDERLINE_SUSPICIOUS_PROMPT,
            "percentage": 15
        },
        "subtle_scam": {
            "prompt": SUBTLE_SCAM_PROMPT,
            "percentage": 10
        }
    }
}

def get_prompt_for_category(category: str) -> str:
    """Get the appropriate prompt for a given category"""
    category_prompts = {
        "authority_scam": AUTHORITY_SCAM_PROMPT,
        "tech_scam": TECH_SUPPORT_SCAM_PROMPT,
        "legitimate_half_a": LEGITIMATE_CUSTOMER_SERVICE_PROMPT,
        "urgency_scam": URGENCY_SCAM_PROMPT,
        "financial_scam": FINANCIAL_FRAUD_PROMPT,
        "legitimate_half_b": LEGITIMATE_PERSONAL_BUSINESS_PROMPT,
        "borderline": BORDERLINE_SUSPICIOUS_PROMPT,
        "subtle_scam": SUBTLE_SCAM_PROMPT
    }
    
    return category_prompts.get(category, LEGITIMATE_CUSTOMER_SERVICE_PROMPT)

def get_model_config(model_type: str) -> dict:
    """Get configuration for a specific model type"""
    if model_type == "A":
        return MODEL_A_CONFIG
    elif model_type == "B":
        return MODEL_B_CONFIG
    else:
        raise ValueError(f"Unknown model type: {model_type}") 
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the FAQ dataset
faqs = {
    "Does Rustafied have a Discord?": "Yes! Rustafied does have a Discord where you can find wipe times, map voting, server information, and more. Join here: discord.gg/rustafied",
    "How can I join the Rustafied team?": "All positions at Rustafied are volunteer. You can apply here: www.rustafied.com/apply",
    "How can I get help?": "For ban appeals and VIP support, submit a ticket here: www.rustafied.com/support. To report players, visit https://my.rustafied.com or type /report in-game or on Discord.",
    "How do I contact Bugs?": "Submit a support ticket here: www.rustafied.com/support. Bugs is also in our Discord, but his DMs are closed.",
    "Can I transfer my VIP slot to a different server/player?": "We do not offer transfers. If you want access to a different Rustafied server, you need to purchase VIP for that server.",
    "I entered my SteamID incorrectly, help!": "Submit a support ticket with your correct Steam ID 64 so we can fix this for you.",
    "When will VIP be restocked?": "VIP is restocked as players cancel or do not renew their subscriptions. Check back periodically as we update our stock frequently.",
    "Can you increase the VIP slots on the server?": "VIP slots are limited to maintain balance and server performance.",
    "Can I get a refund?": "VIP is a digital service and is non-refundable. You can cancel your subscription at any time.",
    "Is VIP a monthly subscription?": "Yes, VIP is a monthly subscription that you can cancel anytime. You retain VIP access for the period you paid for.",
    "How do I cancel my renewals?": "Go to Manage Purchases, click 'Manage' next to the VIP slot, and cancel your subscription.",
    "Can I make a one-time payment to skip queue?": "No, we do not offer one-time payments to skip the queue. VIP is a monthly subscription.",
    "I bought VIP and it isn’t working.": "Exit the queue and rejoin. Ensure you are connecting to the correct server/region. If issues persist, submit a ticket: www.rustafied.com/support",
    "How do I activate my VIP?": "VIP activates automatically after purchase. If queue skipping does not work, wait a minute and retry. If issues persist, submit a support ticket.",
    "I was banned and have VIP. Can I get a refund?": "VIP is non-refundable. VIP does not exempt players from server rules.",
    "How do I report a player?": "Submit a report at https://my.rustafied.com or type /report in-game or in Discord.",
    "Why haven’t you banned the hacker yet?": "Moderators need sufficient evidence to ban a player. Submit reports at https://my.rustafied.com.",
    "Does Rustafied allow VAC banned players?": "VAC banned players are allowed if their bans are over 180 days old.",
    "Can I appeal a ban for a friend?": "No, only the banned individual can submit an appeal at www.rustafied.com/support.",
    "How long does it take to get a response to my support ticket?": "We aim to respond within 24-48 hours, but weekends and holidays may extend this timeframe.",
    "When is wipe?": "Various servers wipe at different times. Check the detailed schedule on our Discord: www.rustafied.com/discord",
    "When is force wipe?": "Facepunch releases a forced wipe on the first Thursday of each month, usually around 2 PM EST.",
    "When do blueprints wipe?": "Blueprints wipe every month except for US/EU Medium 3 and US/EU Long 3, unless forced by Facepunch.",
    "Can you restart the server? It's lagging!": "Restarting does not always fix lag. You can report server issues on Discord or submit a support ticket.",
    "How do I report admin abuse?": "Send an email to rustafied@gmail.com. Your complaint will be investigated internally." 
}

# Encode FAQ questions as embeddings
faq_questions = list(faqs.keys())
faq_answers = list(faqs.values())
faq_embeddings = model.encode(faq_questions)

# Create a FAISS index
d = faq_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(np.array(faq_embeddings))

def get_answer(query):
    """Find the most relevant FAQ answer for a user query."""
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, 1)  # Get the closest match
    best_match_idx = I[0][0]
    return faq_answers[best_match_idx] if D[0][0] < 5 else "Sorry, please contact a admin/mod to get your question answer"  # Adjust threshold

# Example usage
user_query = "How do I join the Rustafied Discord?"
response = get_answer(user_query)
print(response)

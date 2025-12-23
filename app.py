# app.py (Offline RAG with Intent Routing + Greetings)

import pickle
import faiss
from sentence_transformers import SentenceTransformer

# ---------------------------
# LOAD FAISS INDEX & DATA
# ---------------------------
with open("faiss_paragraphs.pkl", "rb") as f:
    paragraphs = pickle.load(f)

index = faiss.read_index("faiss_index.idx")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------
# IN-MEMORY APPOINTMENTS
# ---------------------------
appointments = []

# ---------------------------
# RETRIEVAL FUNCTION
# ---------------------------
def retrieve(query, top_k=3):
    q_emb = embed_model.encode([query])
    distances, indices = index.search(q_emb, top_k)
    return [paragraphs[i] for i in indices[0]]

# ---------------------------
# MAIN CHAT FUNCTION
# ---------------------------
def elitebody_chat(query):
    q = query.lower().strip()

    # ---- GREETING INTENT ----
    if q in ["hi", "hello", "hey", "good morning", "good evening"]:
        return (
            " Hello! Welcome to **Elite Body Home Polyclinic**.\n\n"
            "How can I help you today?\n"
            "• Treatments & services\n"
            "• Working hours\n"
            "• Location & contact details\n"
            "• Book an appointment"
        )

    # ---- BASIC DETAILS ----
    if any(word in q for word in ["about", "clinic", "elite body home", "who are you"]):
        return (
            " **Elite Body Home Polyclinic** is a beauty and wellness clinic based in Dubai.\n\n"
            "We provide high-quality, non-invasive aesthetic and medical wellness treatments "
            "delivered by experienced, DHA-certified doctors in a welcoming environment.\n\n"
            "We focus on personalized care using advanced technology and ISO international "
            "quality standards."
        )

    # ---- WORKING HOURS ----
    if any(word in q for word in ["working hours", "timings", "hours", "open"]):
        return " **Working Hours:** Monday to Sunday, 9:00 AM to 9:00 PM."

    # ---- LOCATION ----
    if any(word in q for word in ["location", "address", "where"]):
        return (
            "**Our Location:**\n"
            "2nd December Street, Jumeirah 1,\n"
            "Al Hudaiba Awards Buildings Block B,\n"
            "1st Floor, Dubai."
        )

    # ---- CONTACT DETAILS ----
    if any(word in q for word in ["contact", "phone", "email", "call"]):
        return (
            " **Contact Information:**\n"
            "Email: contact@elitebodyhome.com\n"
            "Phone: +971 55 120 0086\n"
            "Phone: +971 4 547 9492"
        )

    # ---- TREATMENTS / SERVICES ----
    if any(word in q for word in ["treatment", "treatments", "services", "procedures"]):
        return (
            " **Services & Treatments at Elite Body Home**\n\n"
            "We offer a wide range of non-surgical aesthetic and wellness treatments:\n"
            "- Body sculpting\n"
            "- Cryolipolysis (non-invasive fat freezing)\n"
            "- Aqualyx fat dissolving treatment\n"
            "- Skin tightening\n"
            "- Cellulite reduction\n"
            "- Laser treatments\n"
            "- Dermatology services\n"
            "- Slimming treatments\n"
            "- Physiotherapy\n"
            "- IV therapy\n\n"
            " **Highlights:**\n"
            "• Cryolipolysis targets stubborn fat without surgery or downtime.\n"
            "• Aqualyx eliminates localized fat with minimal downtime."
        )

    # ---- APPOINTMENT BOOKING ----
    if "book" in q or "appointment" in q:
        name = input("Enter your name: ")
        treatment = input("Preferred treatment: ")
        date = input("Preferred date: ")
        time = input("Preferred time: ")

        appointments.append({
            "name": name,
            "treatment": treatment,
            "date": date,
            "time": time
        })

        return (
            f"**Appointment Booked Successfully!**\n"
            f"Name: {name}\n"
            f"Treatment: {treatment}\n"
            f"Date: {date}\n"
            f"Time: {time}"
        )

    # ---- FALLBACK: FAISS SEMANTIC SEARCH ----
    results = retrieve(query, top_k=3)
    if results:
        return "\n\n".join(results)

    return "Sorry, I couldn't find relevant information. Please try asking differently."

if __name__ == "__main__":
    print("Elite Body Home Polyclinic Chatbot (Offline RAG)")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break

        response = elitebody_chat(user_input)
        print("\nBot:", response)

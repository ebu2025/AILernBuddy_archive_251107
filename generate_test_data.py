import requests
import json
import time
import random

BASE_URL = "http://127.0.0.1:8000"

# Question pools by topic
QUESTIONS = {
    "bpmn": [
        "What is BPMN?",
        "Explain swimlanes in BPMN",
        "What is an XOR gateway?",
        "How do you model parallel processes?",
        "What are BPMN events?",
        "Describe sequence vs message flow.",
        "Explain AND vs XOR gateways.",
        "Model an online order process.",
        "Describe complex BPMN subprocesses."
    ],
    "mathematics": [
        "Define a quadratic function.",
        "What is a parabola?",
        "Explain effects of parameters a,b,c.",
        "Solve f(x)=x^2-4x+3 for zeros.",
        "Explain vertex form derivation.",
        "Application of quadratic functions in optimization."
    ],
    "language_de_en": [
        "What is a metaphor?",
        "Define colloquial language.",
        "Explain 'to break the ice'.",
        "Distinguish between written and spoken language.",
        "Analyze metaphors in political speeches.",
        "Assess the impact of language on society."
    ],
    "language_zh_en": [
        "What does 你好 mean?",
        "How do you greet in Chinese?",
        "Compare sentence structure between German and Chinese.",
        "Explain the significance of tones in Mandarin.",
        "Discuss challenges in learning Mandarin.",
        "Application of Chinese grammar in complex sentences."
    ]
}

USERS = {
    "alice": {
        "bpmn": "K1",
        "mathematics": "K3",
        "language_de_en": "K5",
        "language_zh_en": "K2"
    },
    "peter": {
        "bpmn": "K2",
        "mathematics": "K4",
        "language_de_en": "K6",
        "language_zh_en": "K3"
    },
    "marco": {
        "bpmn": "K6",
        "mathematics": "K5",
        "language_de_en": "K2",
        "language_zh_en": "K1"
    }
}

def test_connection():
    try:
        r = requests.get(BASE_URL)
        print(f"Server status: {r.status_code}")
        return r.status_code == 200
    except Exception as e:
        print(f"Server connection error: {e}")
        return False

def user_exists(user_id):
    try:
        r = requests.get(f"{BASE_URL}/user/{user_id}")
        return r.status_code == 200
    except:
        return False

def register_user(user_id, password="test123"):
    if user_exists(user_id):
        print(f"User {user_id} already exists.")
        return True
    try:
        r = requests.post(f"{BASE_URL}/auth/register", json={
            "user_id": user_id,
            "email": f"{user_id}@test.com",
            "password": password
        })
        if r.status_code == 200:
            print(f"User {user_id} registered successfully.")
            return True
        else:
            print(f"Registration failed for {user_id}: {r.status_code}")
            return False
    except Exception as e:
        print(f"Exception during registration for {user_id}: {e}")
        return False

def simulate_chat(user_id, topic, iterations):
    question_pool = QUESTIONS.get(topic, [])
    success_count = 0
    for i in range(iterations):
        question = random.choice(question_pool)
        # 80% chance simulate correct answer
        is_correct = random.random() < 0.8
        answer = "correct answer simulated" if is_correct else "incorrect answer simulated"

        print(f"[{user_id}][{topic}] Question: {question}")
        resp = requests.post(f"{BASE_URL}/chat", json={
            "user_id": user_id,
            "topic": topic,
            "text": question,
            "apply_mode": "auto"
        })
        if resp.ok:
            print(" → Question received successfully")
        else:
            print(f" → Error: {resp.status_code}")

        time.sleep(0.3)

        resp = requests.post(f"{BASE_URL}/chat", json={
            "user_id": user_id,
            "topic": topic,
            "text": answer,
            "apply_mode": "auto"
        })
        if resp.ok:
            print(" → Microcheck answer received")
            success_count += 1
        else:
            print(f" → Microcheck error: {resp.status_code}")

        time.sleep(0.3)

    print(f"{success_count} of {iterations} chats successful.")

def run_tests():
    if not test_connection():
        return

    # Register users if needed
    for user in USERS.keys():
        register_user(user)

    total_tests = 0
    iterations_per_topic = 5

    for user, levels in USERS.items():
        for topic in QUESTIONS.keys():
            level = levels.get(topic, "K1")
            print(f"\nRunning tests for {user} at level {level} on topic {topic}")
            simulate_chat(user, topic, iterations_per_topic)
            total_tests += iterations_per_topic

    print(f"\nTotal tests run: {total_tests}")

    # Check analytics after tests
    r = requests.get(f"{BASE_URL}/teacher/analytics?limit=200")
    if r.ok:
        analytics = r.json()
        print(f"Analytics Entries: {len(analytics)}")
    else:
        print("Failed to get analytics")

if __name__ == "__main__":
    run_tests()

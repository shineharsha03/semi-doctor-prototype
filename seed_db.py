import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))

def seed_database():
    driver = GraphDatabase.driver(URI, auth=AUTH)
    print("🧠 Teaching the AI Medical Knowledge...")
    
    # We create Diseases, Symptoms, and links between them
    query = """
    MERGE (malaria:Disease {name: "Malaria"})
    MERGE (flu:Disease {name: "Flu"})
    MERGE (covid:Disease {name: "Covid-19"})
    MERGE (lupus:Disease {name: "Lupus"})
    
    MERGE (fever:Symptom {name: "Fever"})
    MERGE (chills:Symptom {name: "Chills"})
    MERGE (fatigue:Symptom {name: "Fatigue"})
    MERGE (rash:Symptom {name: "Butterfly Rash"})
    MERGE (cough:Symptom {name: "Dry Cough"})
    
    MERGE (malaria)-[:CAUSES]->(fever)
    MERGE (malaria)-[:CAUSES]->(chills)
    MERGE (flu)-[:CAUSES]->(fever)
    MERGE (flu)-[:CAUSES]->(fatigue)
    MERGE (covid)-[:CAUSES]->(fever)
    MERGE (covid)-[:CAUSES]->(cough)
    MERGE (lupus)-[:CAUSES]->(fatigue)
    MERGE (lupus)-[:CAUSES]->(rash)
    """
    
    with driver.session() as session:
        session.run(query)
        
    driver.close()
    print("✅ Knowledge Graph Created Successfully!")

if __name__ == "__main__":
    seed_database()
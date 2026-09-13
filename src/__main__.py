import json
from llm_sdk import Small_LLM_Model
from .decoder import generate_json


def main():
    print("🤖 Chargement du modèle en mémoire...")
    llm = Small_LLM_Model()

    vocab_path = llm.get_path_to_vocab_file()
    print("🔍 Chargement du vocabulaire complet...")
    with open(vocab_path) as f:
        full_vocab = json.load(f)

    # 1. Charger les fonctions pour extraire automatiquement les noms valides
    with open("data/input/functions_definition.json") as f:
        functions_data = json.load(f)
    valid_names = [func["name"] for func in functions_data]

    # 2. Charger ton fichier de prompts (adapte le nom du fichier si besoin, ex: prompts.json)
    with open("data/input/function_calling_tests.json") as f:
        prompts_data = json.load(f)

    print(f"\n🚀 Lancement de la génération pour les {len(prompts_data)} tests...\n")

    # 3. Boucler sur chaque prompt du fichier
    for i, item in enumerate(prompts_data):
        user_query = item["prompt"]
        print(f"[{i+1}/{len(prompts_data)}] Requête : {user_query}")

        # Contexte d'amorce amélioré pour aider le petit modèle à choisir la bonne fonction
        prompt = (
            f"Available functions:\n{json.dumps(functions_data, indent=2)}\n\n"
            f"User Query: \"{user_query}\"\n"
            f"Generate the strict JSON response:\n"
        )

        # Lancement de la génération sous contrainte pour cette requête
        generate_json(llm, prompt, full_vocab, valid_names)
        print("\n" + "-"*40)

    print("\n✅ Tous les tests sont terminés !")


if __name__ == "__main__":
    main()
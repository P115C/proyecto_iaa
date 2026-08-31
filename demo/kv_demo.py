from openai import OpenAI

from iaa_memoria_1 import KVExtractor, KVStore, VectorStoreIndex

"""
KVStore con deduplicación de claves mediante índice vectorial.
- Usa un esquema fijo de categorías para determinismo.
- Mantiene un índice de embeddings de las claves existentes.
- Al añadir una nueva clave, busca si ya existe una semánticamente similar.
- Si la similitud supera el umbral, reutiliza la clave existente.
- Persistencia en JSON para datos y en ChromaDB para el índice de claves.
"""


if __name__ == "__main__":
    # 1. Inicializar componentes
    print("-- Inicializando KVStore con índice de claves...")

    # Índice persistente (los embeddings se guardan en ./chroma_db)
    key_index = VectorStoreIndex()
    kv = KVStore("kv_store_demo.json", key_index=key_index, dedup_threshold=0.80)

    # Cliente LLM (necesitas tu API key)
    openai_client = OpenAI(api_key="TU_API_KEY")
    extractor = KVExtractor(openai_client, user_id="user_123")

    # 2. Primera interacción: el usuario da su información
    print("\n  - Texto 1: 'Soy Ana, vivo en Madrid y mi correo es ana@correo.com'")
    text1 = "Soy Ana, vivo en Madrid y mi correo es ana@correo.com"
    facts1 = extractor.extract_facts(text1)
    print("Hechos extraídos:", facts1)

    for fact in facts1:
        suggested_key = extractor.build_key(
            fact["entity_type"], fact["entity_id"], fact["category"]
        )
        final_key = kv.set(suggested_key, fact["value"])
        print(f"  - Guardado: '{final_key}' = '{fact['value']}'")

    # 3. Segunda interacción: el usuario dice casi lo mismo, pero con otra redacción
    print("\n  - Texto 2: 'Mi email es ana@gmail.com y mi ciudad es Madrid'")
    text2 = "Mi email es ana@gmail.com y mi ciudad es Madrid"
    facts2 = extractor.extract_facts(text2)
    print("Hechos extraídos:", facts2)

    for fact in facts2:
        suggested_key = extractor.build_key(
            fact["entity_type"], fact["entity_id"], fact["category"]
        )
        final_key = kv.set(suggested_key, fact["value"])
        print(f"  - Guardado: '{final_key}' = '{fact['value']}'")

    # 4. Ver el estado final del almacén
    print("\n-- Estado completo del KVStore:")
    for k, v in kv.get_all().items():
        print(f"  {k} → {v}")

    # 5. Comprobar que las claves se han deduplicado
    print("\n-- Claves en el índice vectorial:")
    all_keys = key_index.get_all_keys()
    for idx, k in enumerate(all_keys, 1):
        print(f"  {idx}. {k}")

    # 6. Demostración de recuperación
    print("\n-- Recuperación:")
    email_key = extractor.build_key("user", "user_123", "contact_email")
    email = kv.get(email_key)
    print(f"  Email: {email}")

    location_key = extractor.build_key("user", "user_123", "location")
    location = kv.get(location_key)
    print(f"  Ubicación: {location}")

    # 7. Limpieza (opcional)
    # key_index.reset()  # Descomentar para borrar el índice

import json
import re

from openai import OpenAI


class KVExtractor:
    """
    Extractor de hechos para el KVStore.
    Utiliza un LLM para extraer hechos y asignarles una categoría canónica:
    usa una lista de categorías preferidas si aplican, y si no, propone una
    categoría nueva normalizada (minúsculas, snake_case, un solo concepto).
    """

    # Vocabulario preferido: guía para el LLM, no una lista cerrada.
    PREFERRED_CATEGORIES = (
        "location",
        "contact_email",
        "phone_number",
        "employer",
        "job_title",
        "full_name",
        "preference",
        "project_name",
        "tool_name",
        "other",
    )

    def __init__(
        self,
        llm_client: OpenAI,
        user_id: str = "current_user",
    ):
        self.llm = llm_client
        self.user_id = user_id

    @staticmethod
    def normalize_category(category: str) -> str:
        """
        Normaliza una categoría propuesta por el LLM:
        minúsculas, palabras separadas por guion bajo, sin espacios ni símbolos.
        """
        cat = category.strip().lower()
        cat = re.sub(r"[^a-z0-9]+", "_", cat)
        cat = cat.strip("_")
        return cat

    @staticmethod
    def is_valid_category(category: str) -> bool:
        """Devuelve True si la categoría es una clave canónica válida."""
        return bool(re.fullmatch(r"[a-z][a-z0-9_]{0,63}", category))

    def extract_facts(
        self,
        raw_text: str,
    ) -> list[dict[str, str]]:
        prompt = f"""
        Eres un extractor de atributos para un sistema de memoria.
        Dado el siguiente texto del usuario, extrae TODOS los hechos que puedan representarse como atributos de una entidad.

        Categorías preferidas (úsalas SI aplican):
        {", ".join(self.PREFERRED_CATEGORIES)}

        Si ninguna de las categorías preferidas encaja, propón una categoría NUEVA siguiendo estas reglas:
        - En minúsculas y en snake_case (palabras separadas por guion bajo).
        - Un ÚNICO concepto (por ejemplo "food_allergy", "favorite_music_genre", "family_member").
        - Sin artículos, sin preposiciones y sin frases: solo el sustantivo/adjetivo que define el atributo.

        Para cada hecho, devuelve un objeto JSON con:
        - "entity_type": el tipo de entidad (user, project, organization, location, tool). Usa "user" por defecto.
        - "entity_id": el identificador de la entidad. Si el usuario se refiere a sí mismo, usa "{self.user_id}". Si menciona otro nombre, úsalo.
        - "category": la categoría canónica (una de las preferidas si aplica, o una nueva en snake_case).
        - "value": el valor normalizado (limpio, sin artículos, en formato canónico).

        Reglas:
        - Si el texto no contiene información que encaje en alguna categoría, devuelve un array vacío.
        - No incluyas explicaciones, solo el JSON.

        Texto del usuario: "{raw_text}"
        """
        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Eres un extractor de atributos. Devuelve solo JSON válido.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        result = json.loads(response.choices[0].message.content)
        facts = result.get("facts", [])
        valid_facts = []
        for fact in facts:
            category = self.normalize_category(fact.get("category", ""))
            if not self.is_valid_category(category):
                continue
            fact["category"] = category
            if not fact.get("entity_id"):
                fact["entity_id"] = self.user_id
            if not fact.get("entity_type"):
                fact["entity_type"] = "user"
            valid_facts.append(fact)
        return valid_facts

    def build_key(
        self,
        entity_type: str,
        entity_id: str,
        category: str,
    ) -> str:
        cat = self.normalize_category(category)
        return f"{entity_type}:{entity_id}:{cat}"

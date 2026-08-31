from datetime import UTC, datetime

import chromadb
from sentence_transformers import SentenceTransformer


class VectorStoreIndex:
    """
    Índice vectorial para claves del KVStore.
    Busca, dentro de la MISMA entidad, una clave existente cuya categoría sea
    semánticamente similar, para evitar duplicados. Los embeddings se calculan
    sobre la categoría (no sobre la clave completa) para que los IDs de entidad
    no diluyan la señal semántica ni fusionen entidades distintas.
    """

    def __init__(
        self,
        collection_name: str = "kv_key_index",
        persist_directory: str = "./chroma_db",
    ):
        """
        Inicializa el índice con ChromaDB persistente.
        """
        # Modelo multilingüe: las categorías se escriben en español.
        self.embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        # Cliente persistente (los datos se guardan en disco)
        self.client = chromadb.PersistentClient(path=persist_directory)
        # Eliminar colección existente si queremos empezar limpio (opcional)
        # self.client.delete_collection(collection_name)  # Descomentar para reset
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Usamos distancia coseno
        )

    @staticmethod
    def _split_key(key: str) -> tuple[str, str]:
        """
        Separa la clave en su parte de entidad y su categoría.
        La comparación semántica se hace SOLO sobre la categoría, de modo que
        claves de entidades distintas nunca se fusionen entre sí.
        """
        if ":" in key:
            entity_part, category = key.rsplit(":", 1)
            return entity_part, category
        return "", key

    def _embed(self, text: str) -> list[float]:
        """Genera el embedding de un texto (la categoría canónica)."""
        return self.embedder.encode(text, convert_to_tensor=False).tolist()

    def find_similar_key(
        self,
        key: str,
        threshold: float = 0.85,
    ) -> tuple[str, float] | None:
        """
        Busca una clave existente de la MISMA entidad cuya categoría sea
        semánticamente similar. Retorna (clave_existente, similitud) si la
        similitud supera el umbral; en caso contrario, None.
        """
        # Si el índice está vacío, no hay nada que buscar
        if self.collection.count() == 0:
            return None

        entity_part, category = self._split_key(key)
        embedding = self._embed(category)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=1,
            where={"entity_part": entity_part},
            include=["distances", "metadatas"],
        )

        if results["ids"] and len(results["ids"][0]) > 0:
            # ChromaDB devuelve distancia (cosine). Similitud = 1 - distancia
            distance = results["distances"][0][0]
            similarity = 1 - distance
            if similarity >= threshold:
                existing_key = results["ids"][0][0]
                return existing_key, similarity
        return None

    def add_key(
        self,
        key: str,
    ) -> None:
        """
        Añade una nueva clave al índice (si no existe ya).
        Guarda en los metadatos la parte de entidad para poder filtrar la
        búsqueda por entidad y evitar fusiones entre entidades distintas.
        """
        # Verificar si ya existe exactamente (para no duplicar)
        existing = self.collection.get(ids=[key])
        if existing["ids"]:
            return

        entity_part, category = self._split_key(key)
        embedding = self._embed(category)
        self.collection.add(
            ids=[key],
            embeddings=[embedding],
            metadatas=[
                {
                    "key": key,
                    "entity_part": entity_part,
                    "created_at": datetime.now(UTC).isoformat(),
                }
            ],
        )

    def get_all_keys(self) -> list[str]:
        """Devuelve todas las claves almacenadas en el índice."""
        if self.collection.count() == 0:
            return []
        results = self.collection.get(include=["ids"])
        return results.get("ids", [])

    def reset(self) -> None:
        """Elimina todos los datos del índice (útil para pruebas)."""
        all_ids = self.get_all_keys()
        if all_ids:
            self.collection.delete(ids=all_ids)

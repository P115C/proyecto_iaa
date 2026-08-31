import json
from datetime import UTC, datetime

from VectorStoreIndex import VectorStoreIndex


class KVStore:
    """
    Almacén clave-valor persistente con deduplicación semántica de claves.
    """

    def __init__(
        self,
        filepath: str = "kv_store.json",
        key_index: VectorStoreIndex | None = None,
        dedup_threshold: float = 0.85,
    ):
        self.filepath = filepath
        self.dedup_threshold = dedup_threshold
        # Si no se proporciona índice, creamos uno por defecto
        self.key_index = key_index if key_index else VectorStoreIndex()
        self.data: dict = {}
        self._load()
        # Sincronizar el índice con los datos existentes (para evitar inconsistencias)
        self._sync_index_with_data()

    def _sync_index_with_data(self) -> None:
        """Asegura que todas las claves en 'data' estén en el índice."""
        existing_keys = set(self.data.keys())
        indexed_keys = set(self.key_index.get_all_keys())
        missing = existing_keys - indexed_keys
        for key in missing:
            self.key_index.add_key(key)

    def set(
        self,
        key: str,
        value: str,
    ) -> str:
        """
        Guarda un valor bajo una clave.
        Primero intenta deduplicar: si existe una clave similar, la reutiliza.
        Retorna la clave final utilizada (puede ser la original o la existente).
        """
        # 1. Buscar clave similar en el índice
        similar = self.key_index.find_similar_key(key, self.dedup_threshold)
        if similar:
            existing_key, similarity = similar
            # Reutilizar la clave existente
            key = existing_key
            print(f"-- Deduplicación: '{key}' (similitud {similarity:.2f})")
        else:
            # 2. Si no hay similar, añadir la nueva clave al índice
            self.key_index.add_key(key)

        # 3. Escribir/sobrescribir el valor
        self.data[key] = {"value": value, "updated_at": datetime.now(UTC).isoformat()}
        self._save()
        return key

    def get(self, key: str) -> str | None:
        """Devuelve el valor actual para la clave exacta."""
        entry = self.data.get(key)
        return entry["value"] if entry else None

    def search_by_value(self, value: str) -> list[tuple[str, str]]:
        """
        Búsqueda exacta por valor (útil para depuración).
        Retorna lista de (clave, valor).
        """
        results = []
        for k, v in self.data.items():
            if value in v["value"]:  # coincidencia parcial
                results.append((k, v["value"]))
        return results

    def get_all(self) -> dict[str, str]:
        """Devuelve todas las claves y valores actuales."""
        return {k: v["value"] for k, v in self.data.items()}

    def get_metadata(self, key: str) -> dict | None:
        """Devuelve el metadato completo de una clave (incluyendo timestamp)."""
        return self.data.get(key)

    def delete(self, key: str) -> bool:
        """Elimina una clave del almacén y del índice."""
        if key in self.data:
            del self.data[key]
            # Eliminar del índice también
            try:
                self.key_index.collection.delete(ids=[key])
            except Exception:
                pass
            self._save()
            return True
        return False

    def _load(self) -> None:
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except FileNotFoundError:
            self.data = {}

    def _save(self) -> None:
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

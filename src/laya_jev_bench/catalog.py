from dataclasses import dataclass


@dataclass(frozen=True)
class SourceFile:
    path: str
    url: str
    sha256: str


BANKING77_REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
ARBANKING77_REVISION = "2e3a5639e254bc28828ed0af6d3059d64d3b97fc"
CLINC150_REVISION = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
LAYA_VARIANTS = {
    "multilingual": {
        "model": "models/laya-multilingual-151",
        "source": "convaiinnovations/laya-multilingual",
        "revision": "052592a15d198d9ad47da779604259b10b47b7aa",
        "max_length": 1024,
    },
    "typed-decisions": {
        "model": "models/laya-typed-decisions-151",
        "source": "convaiinnovations/laya-typed-decisions",
        "revision": "f9ab0b228f0fc0f14d873dbc99038f135c2da1b2",
        "max_length": 1024,
    },
}
LAYA_DATASET_VARIANTS = {
    "banking77": "typed-decisions",
    "arbanking77": "multilingual",
    "clinc150": "typed-decisions",
}
OPENROUTER_JEV_MODEL = "typesafe/jev-1.13"
TYPESAFE_JEV_MODEL = "jev-1.13.0"
OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
TYPESAFE_SYSTEM_ONE_URL = "https://api.typesafe.ai/v1/systemone"


def github_raw(repository: str, revision: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repository}/{revision}/{path}"


SOURCES = {
    "banking77": (
        SourceFile(
            "LICENSE",
            github_raw("PolyAI-LDN/task-specific-datasets", BANKING77_REVISION, "LICENSE"),
            "7e7170e3cebf88a9f60c7b8421418323c09304da1af4d5e90f4da1dc1c8a2661",
        ),
        SourceFile(
            "categories.json",
            github_raw(
                "PolyAI-LDN/task-specific-datasets",
                BANKING77_REVISION,
                "banking_data/categories.json",
            ),
            "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63",
        ),
        SourceFile(
            "test.csv",
            github_raw(
                "PolyAI-LDN/task-specific-datasets",
                BANKING77_REVISION,
                "banking_data/test.csv",
            ),
            "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
        ),
    ),
    "arbanking77": (
        SourceFile(
            "LICENSE",
            github_raw("SinaLab/ArBanking77", ARBANKING77_REVISION, "LICENSE"),
            "2c9f00c84d0681f06ce5ccb0dfbccea531e3850f0f14a99561feb316f4a0fdff",
        ),
        SourceFile(
            "intents.csv",
            github_raw("SinaLab/ArBanking77", ARBANKING77_REVISION, "data/Banking77_intents.csv"),
            "aaba4ba1e0f8471e4594c1e2c6122d3b97f864bda7b587b360e6a41f44269c93",
        ),
        SourceFile(
            "msa.csv",
            github_raw(
                "SinaLab/ArBanking77",
                ARBANKING77_REVISION,
                "data/Banking77_Arabized_MSA_test.csv",
            ),
            "2e72b0272a06c9d55afd8ceec562b6a11348602e28aceed1676dae2cdb16ec9b",
        ),
        SourceFile(
            "palestinian.csv",
            github_raw(
                "SinaLab/ArBanking77",
                ARBANKING77_REVISION,
                "data/Banking77_Arabized_PAL_test.csv",
            ),
            "6d80436bf89c8257894baa178a34f090c01975671d33dfd83da5eb2501341a43",
        ),
        SourceFile(
            "saudi.csv",
            github_raw(
                "SinaLab/ArBanking77",
                ARBANKING77_REVISION,
                "data/Banking77_Arabized_Saudi_test.csv",
            ),
            "9c01583810f37152a734197c3bf80b6cf9c6d129be25c9e1eac27bc0cd91af74",
        ),
        SourceFile(
            "moroccan.csv",
            github_raw(
                "SinaLab/ArBanking77",
                ARBANKING77_REVISION,
                "data/Banking77_Arabized_Moroccan_test.csv",
            ),
            "1c4063a2af8bc415a31d6ec53b18e23080fdd4814f10467f6b1819d827fa8de8",
        ),
        SourceFile(
            "tunisian.csv",
            github_raw(
                "SinaLab/ArBanking77",
                ARBANKING77_REVISION,
                "data/Banking77_Arabized_Tunisian_test.csv",
            ),
            "1c7c7aded5b5a90e20608a2dd2f3435683f6090d722e284e6e466eee402ef7a0",
        ),
    ),
    "clinc150": (
        SourceFile(
            "LICENSE",
            github_raw("clinc/oos-eval", CLINC150_REVISION, "LICENSE"),
            "e6bc9e9c474700b708f568bac9e5a8a9bcb2b1dad53442f5ba449fcb848b8e76",
        ),
        SourceFile(
            "data_full.json",
            github_raw("clinc/oos-eval", CLINC150_REVISION, "data/data_full.json"),
            "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0",
        ),
    ),
}

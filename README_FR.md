# Assistant conversationnel

Ce dépôt contient l’application de départ pour un assistant conversationnel. Il fournit la structure de l’application, les outils d’évaluation, les scénarios, les tests et la documentation technique de référence nécessaires pour mener le projet à bien. **Les consignes du devoir et les livrables requis sont fournis sur le site du cours**.

La plupart de votre travail d’implémentation se fera dans `assistant/context/`.

```text
assistant/
    ...
    context/
        ...
        full_history.py     --> VOTRE IMPLÉMENTATION
        recent_window.py    --> VOTRE IMPLÉMENTATION
        managed.py          --> VOTRE IMPLÉMENTATION

scenarios/                  Scénarios d’évaluation
logs/                       Traces et résultats d’évaluation
...
config.toml                 Paramètres du projet
```

## Installation

Le projet utilise `uv` pour les environnements Python et les dépendances. Depuis la racine du dépôt, exécutez les commandes suivantes pour installer les dépendances et lancer les tests.

```bash
uv sync
uv run pytest
```

Le projet utilise un serveur Ollama local. Installez Ollama et récupérez le modèle configuré dans `config.toml` :

```bash
ollama pull gemma4:e4b
```

Assurez-vous ensuite qu’Ollama est en cours d’exécution avant de démarrer l’assistant.

Consultez [CONFIGURATION_FR.md](CONFIGURATION_FR.md) pour tous les paramètres de configuration de l’application.

## Exécuter l’assistant

Démarrez l’application avec :

```bash
uv run python -m assistant
```

La politique de contexte utilisée par défaut est configurée dans [config.toml](config.toml). Pour exécuter l’assistant avec une autre politique sans modifier le fichier de configuration, utilisez `--policy` :

```bash
uv run python -m assistant --policy stateless
uv run python -m assistant --policy full_history
uv run python -m assistant --policy recent_window
uv run python -m assistant --policy managed
```

Pendant une conversation, l’interface du terminal prend en charge plusieurs commandes :

```text
Enter         envoyer le message rédigé
Shift+Enter   insérer un saut de ligne

Ctrl+R        démarrer une nouvelle conversation
Ctrl+T        choisir le mode de raisonnement
Ctrl+P        modifier la limite de tokens générés
Ctrl+I        afficher ou masquer l’inspecteur de contexte

Ctrl+Up       restaurer la saisie précédente
Ctrl+Down     avancer dans l’historique des saisies

Ctrl+Q        quitter
```

### Inspection du contexte

Le panneau d’inspection affiche la conversation stockée par l’application ainsi que le contexte sélectionné et envoyé au modèle pour le tour actuel, y compris tout état maintenu par la politique de contexte active. Les sessions interactives enregistrent également la conversation sous forme de trace JSONL dans le répertoire `logs/`. Les résultats d’évaluation contiennent le contexte construit pour le modèle, l’état de la politique et les informations d’utilisation du modèle pour chaque tour.

### Formatage mathématique

Dans les réponses de l’assistant, la syntaxe LaTeX encadrée par `$...$`, `$$...$$`, `\(...\)` ou `\[...\]` est convertie en une approximation Unicode adaptée au terminal. Les signes dollar immédiatement suivis d’un chiffre ne sont pas convertis, ainsi que les blocs de code. L’historique de conversation et les traces JSONL conservent la réponse originale du modèle.

## Exécuter un scénario d’évaluation

Les scénarios situés dans `scenarios/` définissent des conversations reproductibles qui peuvent être exécutées avec différentes politiques de contexte. La commande générale est :

```bash
uv run python -m assistant.evaluation.run \
    scenarios/<scenario>.yaml \
    --policy <policy> \
    --output logs/<output>.json
```

Par exemple :

```bash
uv run python -m assistant.evaluation.run \
    scenarios/recall.yaml \
    --policy full_history \
    --output logs/recall-full-history.json
```

Le fichier JSON généré enregistre les messages exacts envoyés à Ollama, les messages sources sélectionnés par la politique de contexte, l’utilisation du modèle, l’utilisation du contexte et tout état maintenu par la politique.

### Afficher les résultats d’évaluation

Ouvrez [`visualizer/index.html`](visualizer/index.html) dans un navigateur, puis sélectionnez un ou plusieurs fichiers JSON d’évaluation à partir du répertoire `logs/`. Le visualiseur lit les fichiers localement dans le navigateur ; il ne les télécharge pas ni ne nécessite un serveur.

## Référence technique

Utilisez la documentation du dépôt lorsque vous avez besoin de détails sur l’application elle-même :

- [CONFIGURATION_FR.md](CONFIGURATION_FR.md) décrit les paramètres disponibles dans [config.toml](config.toml).
- [CONTEXT_API_FR.md](CONTEXT_API_FR.md) documente l’interface des politiques de contexte.

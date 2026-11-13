# Référence de configuration

Par défaut, l’application lit ses paramètres depuis `config.toml` dans le répertoire de travail courant. (Utilisez `--config` si vous souhaitez spécifier un autre chemin.) Tous les champs documentés sont obligatoires. Les valeurs inconnues, manquantes ou non valides sont signalées lors du chargement de la configuration.

## `[model]`

Ces paramètres configurent les requêtes vers le point `/api/chat` d’Ollama.

### `name`

Le nom du modèle Ollama.

```toml
name = "gemma4:e4b"
```

Le modèle doit déjà être disponible sur le serveur Ollama. Pour un serveur local, téléchargez-le avec :

```bash
ollama pull gemma4:e4b
```

### `host`

L’URL de base du serveur Ollama, sans `/api/chat`.

```toml
host = "http://localhost:11434"
```

Utilisez `localhost` lorsque Ollama et l’assistant s’exécutent sur la même machine. Pour utiliser un serveur Ollama sur une autre machine, indiquez son nom d’hôte accessible ou son adresse sur le réseau local :

```toml
host = "http://192.168.1.75:11434"
```

Un serveur distant doit être configuré pour écouter sur une interface réseau, et son pare-feu doit autoriser le port TCP `11434`. Vérifiez la connexion depuis le même terminal que celui dans lequel vous exécuterez l’assistant :

```bash
curl http://192.168.1.75:11434/api/version
```

### `num_ctx`

La taille de la fenêtre de contexte demandée à Ollama, en tokens.

```toml
num_ctx = 4096
```

Utilisez une valeur prise en charge par le modèle sélectionné. Les traces indiquent l’utilisation du prompt par rapport à cette valeur.

### `num_predict`

Le nombre maximal de tokens qu’Ollama peut générer pour un appel au modèle.

```toml
num_predict = 500
```

Pendant une session interactive, utilisez `Ctrl+P` pour remplacer cette limite lors des appels suivants. Ce changement ne s’applique qu’à la session en cours et ne modifie pas ce fichier.

Lorsqu’Ollama arrête une réponse parce qu’elle atteint cette limite, il renvoie `done_reason = "length"`. L’interface utilisateur ajoute un avertissement à la réponse, et l’inspecteur de contexte affiche `output limit reached`. Augmentez `num_predict` si la tâche nécessite régulièrement des réponses plus longues, tout en conservant une capacité suffisante dans la fenêtre de contexte pour le prompt.

### `temperature`

Contrôle la variabilité de l’échantillonnage. Utilisez un nombre compris entre `0.0` et `2.0`, bornes incluses.

```toml
temperature = 0.2
```

Des valeurs plus faibles rendent généralement les réponses plus cohérentes ; des valeurs plus élevées les rendent généralement plus variées.

### `seed`

Le seed d’échantillonnage envoyée à Ollama.

```toml
seed = 42
```

Un seed fixe améliore la reproductibilité, sans toutefois garantir des résultats identiques entre différentes versions du modèle ou de l’environnement d’exécution.

### `think`

Contrôle quels appels à Ollama utilisent le raisonnement pour les modèles qui le prennent en charge.

```toml
think = "off"
```

Pendant une session interactive, utilisez `Ctrl+T` pour remplacer le mode de raisonnement pour les appels suivants. Ce remplacement ne s’applique qu’à la session en cours et ne modifie pas ce fichier.

Les valeurs acceptées sont :

- `"off"` : aucun appel n’utilise le raisonnement ;
- `"answers"` : seuls les appels produisant des réponses destinées à l’utilisateur utilisent le raisonnement ;
- `"maintenance"` : seuls les appels de maintenance de la politique utilisent le raisonnement ;
- `"all"` : les appels de réponse et de maintenance utilisent tous deux le raisonnement.

Les appels de maintenance dépendent de la politique. Par exemple, la politique `managed` en effectue un pour mettre à jour son état conservé, tandis que des politiques plus simples n’en effectuent pas. Les modèles non compatibles peuvent ignorer ou refuser le mode de raisonnement.

Lorsque le raisonnement est activé, le modèle peut générer un raisonnement interne supplémentaire avant de produire sa réponse finale. **Les tokens utilisés pour le raisonnement sont comptabilisés dans la limite `num_predict` configurée.** Il est conseillé d’augmenter cette limite lorsque le raisonnement est activé, afin que le modèle dispose de suffisamment de marge pour produire une réponse complète après son raisonnement.

### `timeout_seconds`

La durée maximale, en secondes, pendant laquelle le client HTTP attend une requête Ollama.

```toml
timeout_seconds = 300
```

Utilisez un entier positif suffisamment élevé pour le modèle selectionné. Les échecs de connexion et les délais d’attente dépassés sont signalés comme une impossibilité d’atteindre l’hôte Ollama configuré.

## `[assistant]`

### `system_prompt`

L’instruction système placée au début de chaque contexte du modèle.

```toml
[assistant]
system_prompt = """Vous êtes un assistant d'étude..."""
```

Il doit s’agir d’une chaîne non vide. Modifier le prompt système peut changer de manière significative le comportement du modèle.

## `[context]`

### `policy`

La politique de contexte utilisée par l’assistant interactif.

```toml
policy = "stateless"
```

Les valeurs valides sont :

- `stateless` : le prompt système plus le message actuel de l’utilisateur ;
- `full_history` : l’intégralité de la conversation stockée ;
- `recent_window` : le prompt système, les tours récents terminés et le message actuel de l’utilisateur ;
- `managed` : les tours récents terminés plus un état conversationnel conservé.

L’option `--policy` de l’application remplace ce paramètre pour une exécution. Le lanceur de scénarios exige que `--policy` soit spécifié explicitement.

### `recent_turns`

La taille configurée de la fenêtre des tours récents disponible pour les politiques de contexte.

```toml
recent_turns = 3
```

La valeur est transmise à chaque politique de contexte sous la forme `self.recent_turns` ; elle ne modifie pas à elle seule le contexte du modèle. La politique `recent_window` doit l’utiliser comme nombre maximal de tours utilisateur/assistant terminés à conserver avant le message actuel de l’utilisateur. Une politique `managed` peut également l’utiliser lors de la construction de son contexte.

## `[trace]`

### `directory`

Le répertoire utilisé pour les traces de conversation interactives au format `.jsonl`.

```toml
directory = "logs"
```

Les chemins relatifs sont résolus à partir du répertoire de travail courant du processus, et non à partir de l’emplacement du fichier de configuration. Les répertoires parents sont créés automatiquement.

Les chemins de sortie des évaluations sont sélectionnés indépendamment avec l’option `--output` du lanceur de scénarios.

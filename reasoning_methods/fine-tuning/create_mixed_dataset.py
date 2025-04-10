# create_mixed_dataset.py
import random
from datasets import load_dataset, concatenate_datasets, DatasetDict

# --- Konfiguration ---
OUTPUT_DIR = "reasoning_methods/fine-tuning/mixed_finetuning_dataset" # Ordner zum Speichern des Datensatzes
TARGET_TOTAL_SAMPLES = 300000 # Ungefähre Gesamtgröße des gemischten Datensatzes
VALIDATION_SPLIT_PERCENTAGE = 0.05 # 5% für Validierung

# Definition der Datensätze und wie viele Beispiele wir wollen
# Passe 'target_samples' an, um die Mischung zu steuern
DATASET_CONFIG = [
    {
        "name": "Open-Orca/SlimOrca-Dedup",
        "split": "train",
        "target_samples": int(TARGET_TOTAL_SAMPLES * 0.40), # 40%
        "formatting_func": "format_slimorca",
    },
    {
        "name": "allenai/ai2_arc",
        "subset": "ARC-Challenge",
        "split": "train",
        "target_samples": int(TARGET_TOTAL_SAMPLES * 0.125), # 12.5%
        "formatting_func": "format_arc",
    },
    {
        "name": "commonsense_qa",
        "split": "train",
        "target_samples": int(TARGET_TOTAL_SAMPLES * 0.125), # 12.5%
        "formatting_func": "format_commonsense_qa",
    },
    {
        "name": "gsm8k",
        "subset": "main",
        "split": "train",
        "target_samples": int(TARGET_TOTAL_SAMPLES * 0.10), # 10%
        "formatting_func": "format_gsm8k",
    },
    {
        "name": "meta-math/MetaMathQA",
        "split": "train",
        "target_samples": int(TARGET_TOTAL_SAMPLES * 0.10), # 10%
        "formatting_func": "format_metamathqa",
    },
     {
        "name": "squad",
        "split": "train",
        "target_samples": int(TARGET_TOTAL_SAMPLES * 0.075), # 7.5%
        "formatting_func": "format_squad",
    },
    {
        "name": "truthful_qa",
        "subset": "generation",
        "split": "validation", # Ja, 'validation' als Trainingsdaten hier
        "target_samples": int(TARGET_TOTAL_SAMPLES * 0.075), # 7.5%
        "formatting_func": "format_truthful_qa",
    },
]

# --- Formatierungsfunktionen (Anpassung an ChatML-ähnliches Format) ---
# Zielformat: {'messages': [{'from': 'human', 'value': '...'}, {'from': 'gpt', 'value': '...'}]}

def format_slimorca(example):
    # SlimOrca ist oft schon in einem Konversationsformat
    # Wir nehmen an, es hat 'conversations' mit 'from' ('human'/'gpt') und 'value'
    # Eventuell leichte Anpassung nötig, je nach genauer Struktur
    if 'conversations' in example and isinstance(example['conversations'], list):
         # Behalte nur human/gpt turns, falls andere existieren
        messages = [msg for msg in example['conversations'] if msg.get('from') in ['human', 'gpt']]
        # Stelle sicher, dass es mit human beginnt und mit gpt endet (optional, aber oft sinnvoll)
        if messages and messages[0].get('from') == 'human' and messages[-1].get('from') == 'gpt':
            return {"messages": messages}
    return None # Ignoriere Beispiele, die nicht passen

def format_arc(example):
    question = example['question']
    choices_text = "\n".join([f"{label}. {text}" for label, text in zip(example['choices']['label'], example['choices']['text'])])
    prompt = f"{question}\n\nWähle die korrekte Antwort aus den folgenden Optionen:\n{choices_text}"
    answer = example['answerKey'] # Nur der Buchstabe A, B, C, D etc.
    return {"messages": [{'from': 'human', 'value': prompt}, {'from': 'gpt', 'value': answer}]}

def format_commonsense_qa(example):
    question = example['question']
    choices_text = "\n".join([f"{label}. {text}" for label, text in zip(example['choices']['label'], example['choices']['text'])])
    prompt = f"{question}\n\nWähle die beste Antwort aus den folgenden Optionen:\n{choices_text}"
    answer = example.get('answerKey', '') # Manchmal fehlt der Key? Sicherstellen, dass er da ist.
    # Finde den Text zur Antwort
    try:
        answer_text = example['choices']['text'][example['choices']['label'].index(answer)]
    except (ValueError, IndexError):
        answer_text = answer # Fallback auf den Key selbst
    if not answer: return None # Ignoriere Beispiele ohne Antwortkey
    return {"messages": [{'from': 'human', 'value': prompt}, {'from': 'gpt', 'value': f"{answer}. {answer_text}"}]}


def format_gsm8k(example):
    # Nutzt die Chain-of-Thought Antwort
    question = example['question']
    answer = example['answer'] # Enthält oft "Lösung: ..."
    return {"messages": [{'from': 'human', 'value': question}, {'from': 'gpt', 'value': answer}]}

def format_metamathqa(example):
    # MetaMath hat oft 'query' und 'response'
    question = example['query']
    answer = example['response']
    return {"messages": [{'from': 'human', 'value': question}, {'from': 'gpt', 'value': answer}]}

def format_squad(example):
    # Wir brauchen Kontext, Frage und Antwort
    context = example['context']
    question = example['question']
    # Nimm die erste Antwort, SQuAD kann mehrere haben
    answer = example['answers']['text'][0] if example['answers']['text'] else "Keine Antwort gefunden."
    prompt = f"Kontext:\n{context}\n\nFrage:\n{question}\n\nAntworte basierend auf dem Kontext."
    return {"messages": [{'from': 'human', 'value': prompt}, {'from': 'gpt', 'value': answer}]}

def format_truthful_qa(example):
    # Nutzt Frage und die beste Antwort
    question = example['question']
    answer = example['best_answer']
    return {"messages": [{'from': 'human', 'value': question}, {'from': 'gpt', 'value': answer}]}

# Mapping von Funktionsnamen (Strings) zu tatsächlichen Funktionen
FORMATTERS = {
    "format_slimorca": format_slimorca,
    "format_arc": format_arc,
    "format_commonsense_qa": format_commonsense_qa,
    "format_gsm8k": format_gsm8k,
    "format_metamathqa": format_metamathqa,
    "format_squad": format_squad,
    "format_truthful_qa": format_truthful_qa,
}

# --- Hauptlogik ---
all_formatted_samples = []

print("Starte Download und Formatierung der Datensätze...")

for config in DATASET_CONFIG:
    print(f"Verarbeite {config['name']} ({config.get('subset', 'default')})...")
    try:
        # Lade den Datensatz
        ds = load_dataset(config['name'], name=config.get('subset'), split=config['split'], streaming=False) # Nicht streamen für select

        # Wähle zufällige Beispiele (falls der Datensatz größer ist als benötigt)
        num_available = len(ds)
        target_samples = min(config['target_samples'], num_available)
        if num_available > target_samples:
             indices = random.sample(range(num_available), target_samples)
             ds = ds.select(indices)
        else:
             print(f"Warnung: Konnte nur {num_available} statt der gewünschten {target_samples} Beispiele für {config['name']} laden.")


        # Wende die Formatierungsfunktion an
        formatting_func = FORMATTERS[config['formatting_func']]
        formatted_ds = ds.map(formatting_func, remove_columns=ds.column_names, num_proc=4) # Parallelisierung

        # Filtere Beispiele heraus, bei denen die Formatierung fehlgeschlagen ist (None zurückgegeben hat)
        formatted_ds = formatted_ds.filter(lambda example: example is not None and example.get('messages') is not None)
        # Filtere Beispiele ohne Antwort
        formatted_ds = formatted_ds.filter(lambda example: len(example['messages']) > 0 and example['messages'][-1]['from'] == 'gpt' and example['messages'][-1]['value'])


        print(f"-> {len(formatted_ds)} formatierte Beispiele hinzugefügt.")
        all_formatted_samples.append(formatted_ds)

    except Exception as e:
        print(f"Fehler beim Verarbeiten von {config['name']}: {e}")

# Kombiniere alle formatierten Datensätze
print("\nKombiniere alle Datensätze...")
if not all_formatted_samples:
     raise ValueError("Keine Datensätze konnten erfolgreich geladen und formatiert werden.")

mixed_dataset = concatenate_datasets(all_formatted_samples)

# Mische den kombinierten Datensatz gründlich
print(f"Mische den Datensatz ({len(mixed_dataset)} Beispiele)...")
mixed_dataset = mixed_dataset.shuffle(seed=42)

# Erstelle Train/Validation Split
print(f"Erstelle Train/Validation Split ({100-VALIDATION_SPLIT_PERCENTAGE*100}% / {VALIDATION_SPLIT_PERCENTAGE*100}%)...")
split_dataset = mixed_dataset.train_test_split(test_size=VALIDATION_SPLIT_PERCENTAGE, seed=42)

# Umbenennen für Konsistenz (SFTTrainer erwartet oft 'train' und 'test' oder 'validation')
split_dataset["validation"] = split_dataset.pop("test")

print(f"Finaler Trainings-Split: {len(split_dataset['train'])} Beispiele")
print(f"Finaler Validierungs-Split: {len(split_dataset['validation'])} Beispiele")

# Speichere den Datensatz auf der Festplatte
print(f"Speichere Datensatz nach '{OUTPUT_DIR}'...")
split_dataset.save_to_disk(OUTPUT_DIR)

print("\nFertig! Der gemischte Datensatz wurde erstellt und gespeichert.")
print(f"Du kannst ihn jetzt im SFT-Skript mit `--dataset_name {OUTPUT_DIR}` verwenden.")

file_path = "/Users/rezajabbir/Downloads/news.txt"

try:
    with open(file_path, 'r') as file:
        # Lire le contenu du fichier

        words = file.read().split()
        print(words)
        # Chercher le mot "millions" (indépendamment de la casse)
        if any("millions" in word.lower() for word in words):
            print("Le mot 'millions' a été trouvé dans le fichier.")
        else:
            print("Le mot 'millions' n'a pas été trouvé dans le fichier.")

except FileNotFoundError:
    print(f"Le fichier {file_path} n'a pas été trouvé.")
except Exception as e:
    print(f"Une erreur s'est produite : {e}")
# define function to recognize diet type from ingredients list

import re
import numpy as np

nv_teil = [
    "fleisch", "schinken", "speck", "wurst", "würstchen", "salami", "leberkäse",
    "gelatine", "aspik", "kollagen", "pepsin",
    "fisch", "lachs", "hering", "sardell", "anchov", "makrele", "forelle",
    "kabeljau", "garnele", "krabbe", "hummer", "tintenfisch", "kalmar",
    "surimi", "muschel", "auster",
    "hähnchen", "hühnchen", "geflügel", "schweine", "rinder", "hühnerbrühe",
    "schmalz", "talg", "tierfett", "tierisches fett", "knochen", "innereien",
    "blutplasma", "blutwurst",
    "karmin", "cochenille", "kälberlab", "tierisches lab",
]

# --- NICHT VEGETARISCH: nur als ganzes Wort ---
nv_wort = [
    "rind", "rinder", "schwein", "schweine", "kalb", "kälber", "lamm", "lämmer",
    "huhn", "hühner", "pute", "puten", "truthahn", "ente", "enten", "gans",
    "gänse", "wild", "hase", "kaninchen", "leber", "blut", "rogen",
    "e120", "e441",
    # englisch
    "meat", "beef", "pork", "chicken", "turkey", "lamb", "veal", "fish", "ham",
    "bacon", "lard", "tallow", "gelatin", "shrimp", "tuna", "salmon", "anchovy",
    "sausage", "carmine", "cochineal",
]

# Ausnahmen, die vor der NICHT-VEGETARISCH-Prüfung aus dem Text entfernt werden
ausnahmen_vegetarisch = [
    "butterschmalz",      # ist Butter, also vegetarisch
    "austernpilz", "austernseitling",
    "muschelnudel",
]
    
            

# --- NICHT VEGAN (aber vegetarisch): Teilstrings ---
nvegan_teil = [
    "milch", "sahne", "butter", "käse", "joghurt", "jogurt", "yoghurt", "quark",
    "molke", "kasein", "casein", "laktose", "lactose", "laktalbumin",
    "lactalbumin", "schmand", "kefir", "ghee", "topfen", "obers",
    "eigelb", "eiklar", "eipulver", "vollei", "hühnerei",
    "honig", "bienenwachs", "gelée royale", "gelee royale", "propolis",
    "schellack",
    "mozzarella", "parmesan", "gouda", "emmentaler", "cheddar", "mascarpone",
    "ricotta", "camembert", "gorgonzola", "pecorino", "halloumi",
]

# --- NICHT VEGAN: nur als ganzes Wort ---
nvegan_wort = [
    "ei", "eier", "eiweiß", "rahm", "feta", "brie", "e901", "e904",
    # englisch
    "milk", "cream", "butter", "cheese", "yogurt", "yoghurt", "whey", "egg",
    "eggs", "honey", "beeswax", "lactose", "casein",
]

# Ausnahmen, die vor der NICHT-VEGAN-Prüfung aus dem Text entfernt werden
ausnahmen_vegan = [
    # pflanzliche Milch-/Sahne-/Joghurt-Alternativen
    "kokosmilch", "kokosnussmilch", "hafermilch", "sojamilch", "mandelmilch",
    "reismilch", "nussmilch", "haselnussmilch", "cashewmilch", "dinkelmilch",
    "erbsenmilch", "lupinenmilch", "pflanzenmilch",
    "kokossahne", "hafersahne", "sojasahne", "reissahne", "mandelsahne",
    "sojajoghurt", "kokosjoghurt", "mandeljoghurt", "haferjoghurt",
    # pflanzliche "Butter"
    "kakaobutter", "sheabutter", "karitébutter", "mangobutter", "kokosbutter",
    "erdnussbutter", "nussbutter", "mandelbutter", "cashewbutter",
    "haselnussbutter", "sonnenblumenbutter",
    # sonstiges
    "milchsäure", "honigmelone", 
    # englisch
    "coconut milk", "almond milk", "soy milk", "oat milk", "rice milk",
    "cocoa butter", "shea butter", "peanut butter", "coconut cream",
    "lactic acid",
]

def erkenne_ernaehrungsform_aus_zutatenliste(text):
    # 1. fehlende/zu kurze Texte abfangen -> "Unbekannt"
    if not isinstance(text, str) or len(text) < 2:
        return "Unbekannt"
    # 2. Text normalisieren: kleinschreiben, Unterstriche entfernen
    text = text.lower().replace("_", " ")
    # 3. Spurenhinweise abschneiden (Text vor "spuren von" behalten)
    ingredients = text.split("spuren von")[0]
    # 4. Ausnahmen aus dem Text entfernen oder maskieren
    ingredients_vegetarisch_ausnahme = ingredients
    for ausnahme in ausnahmen_vegetarisch:
        ingredients_vegetarisch_ausnahme = ingredients_vegetarisch_ausnahme.replace(ausnahme, " ")

    ingredients_vegan_ausnahme = ingredients
    for ausnahme in ausnahmen_vegan:
        ingredients_vegan_ausnahme = ingredients_vegan_ausnahme.replace(ausnahme, " ")

    # 5. Wörter der Inhaltsliste als Set extrahieren
    woerter_vegetarisch = set(re.split(r"[^\wäöüß]+", ingredients_vegetarisch_ausnahme))
    woerter_vegan = set(re.split(r"[^\wäöüß]+", ingredients_vegan_ausnahme))


    # 6. Wörter der Liste "nicht_vegetarisch" prüfen -> "Nicht-vegetarisch"
    if any(wort in woerter_vegetarisch for wort in nv_wort) or any(teil in ingredients_vegetarisch_ausnahme for teil in nv_teil):
        return "Nicht-vegetarisch"
    # 7. Wörter der Liste "nicht_vegan" prüfen       -> "Vegetarisch"
    if any(wort in woerter_vegan for wort in nvegan_wort) or any(teil in ingredients_vegan_ausnahme for teil in nvegan_teil):
        return "Vegetarisch"
    # 8. sonst      
    
    return "Vegan"    


def ernaehrungsform_aus_tags(tags):
    if not isinstance(tags, str):
        return np.nan

    tag_liste = tags.lower().split(",")

    if "en:non-vegetarian" in tag_liste:
        return "Nicht-vegetarisch"
    if "en:vegan" in tag_liste:
        return "Vegan"
    if "en:vegetarian" in tag_liste:
        return "Vegetarisch"
    return np.nan


if __name__ == "__main__":
    assert erkenne_ernaehrungsform_aus_zutatenliste("weizenmehl, butterschmalz, salz") == "Vegetarisch"
    assert erkenne_ernaehrungsform_aus_zutatenliste("wasser, kokosmilch, zucker") == "Vegan"
    assert erkenne_ernaehrungsform_aus_zutatenliste("nudeln, austernpilz") == "Vegan"
    assert erkenne_ernaehrungsform_aus_zutatenliste("mehl, schweineschmalz") == "Nicht-vegetarisch"
    assert erkenne_ernaehrungsform_aus_zutatenliste("reis, pigmente") == "Vegan"
    assert erkenne_ernaehrungsform_aus_zutatenliste("mehl, eier, salz") == "Vegetarisch"
    assert erkenne_ernaehrungsform_aus_zutatenliste(float("nan")) == "Unbekannt"
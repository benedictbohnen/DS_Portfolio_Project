"""
diettype.py - Ernährungsform eines Produkts bestimmen
=====================================================

Klassen:  "Vegan" | "Vegetarisch" | "Nicht-vegetarisch" | "Unbekannt"

Das Modul kombiniert zwei Informationsquellen:

1. die Zutatenliste (mehrsprachig: de, en, fr, es, it, nl, pl, pt + einige hr/cs/tr-Begriffe)
2. die OpenFoodFacts-Tags aus `ingredients_analysis_tags`

Grundprinzipien
---------------
* Strengste Evidenz gewinnt: Findet EINE Quelle ein Fleisch-/Fischprodukt, ist das
  Produkt "Nicht-vegetarisch". Findet eine Quelle Milch/Ei/Honig, ist es höchstens
  "Vegetarisch". Das hält die vegane Gruppe sauber.
* "Vegan" braucht POSITIVE Evidenz: entweder das Tag `en:vegan` oder ein Zutatentext,
  in dem wir die Sprache erkennen UND nichts Tierisches finden. Unlesbarer oder
  fremdsprachiger Text ohne Tag-Bestätigung ergibt "Unbekannt", nie "Vegan".
* Widersprüche zwischen Text und Tags werden markiert, statt sie zu verstecken.

Benutzung im Notebook
---------------------
    from diettype import klassifiziere_dataframe

    # Alte Spalten zuerst entfernen, sonst gibt join() einen Fehler (doppelte Spaltennamen):
    df_clean = df_clean.drop(columns=[c for c in df_clean if c.startswith("diet_type")])
    df_clean = df_clean.join(klassifiziere_dataframe(df_clean))   # legt diet_type direkt an

Neue Spalten: diet_type, diet_type_quelle, diet_type_konfidenz,
              diet_type_widerspruch, diet_type_details
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from functools import lru_cache

import pandas as pd

NICHT_VEG, VEGETARISCH, VEGAN, UNBEKANNT = "Nicht-vegetarisch", "Vegetarisch", "Vegan", "Unbekannt"

# =============================================================================
# 1. Normalisierung
# =============================================================================
_TRANSLIT = str.maketrans({
    "ß": "ss", "œ": "oe", "æ": "ae", "ø": "o", "ł": "l", "đ": "d", "ı": "i",
    "’": "'", "‘": "'", "´": "'", "`": "'",
})


def _ascii(text: str) -> str:
    """Kleinschreibung, Umlaute/Akzente weg (ä->a, é->e, ł->l, ß->ss, œ->oe)."""
    text = str(text).lower().translate(_TRANSLIT)
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def normalisiere(text: str) -> str:
    """Normalisierung für Zutatentexte (zusätzlich: E-Nummern vereinheitlichen)."""
    text = _ascii(text).replace("_", " ")
    text = re.sub(r"\be[\s\-]?(\d{3,4}[a-z]?)\b", r"e\1", text)   # "E 120" -> "e120"
    return re.sub(r"\s+", " ", text).strip()


# =============================================================================
# 2. Wortlisten
#    *_TEIL  : Begriff darf irgendwo im Wort stehen (nur für eindeutige Wortstämme)
#    *_WORT  : Begriff muss ein ganzes Wort sein (Regex-Fragmente erlaubt, z. B. "eggs?")
# =============================================================================

# ---- NICHT VEGETARISCH (Fleisch, Fisch, tierische Zusatzstoffe) --------------
NV_TEIL = [
    # Deutsch
    "fleisch", "schinken", "speck", "wurst", "salami", "gelatin", "aspik",
    "fisch", "lachs", "hering", "sardell", "anchov", "makrele", "forelle",
    "kabeljau", "garnele", "krabbe", "hummer", "tintenfisch", "kalmar", "surimi",
    "muschel", "auster", "hähnchen", "hühnchen", "geflügel", "schweine",
    "wildschwein", "rinder", "hühnerbrühe", "schmalz", "talg", "tierfett",
    "tierisches fett", "tierische fette", "knochen", "innereien", "blutplasma",
    "kälberlab", "tierisches lab", "leberkäse",
    # mehrsprachig: Kollagen, Karmin/Cochenille
    "collagen", "kollagen", "kolagen", "colagen", "collageen", "pepsin",
    "karmin", "carmin", "cochenil", "cochinil", "cocciniglia", "cochonilha", "koszenil",
    # Niederländisch
    "vlees",
]

NV_WORT = [
    # Deutsch
    "rind", "rinder", "schwein", "schweine", "kalb", "kalber", "lamm", "lammer",
    "huhn", "huhner", "pute", "puten", "truthahn", "ente", "enten", "gans", "ganse",
    "kaninchen", "leber", "blut", "rogen",
    # Englisch
    "meat", "beef", "pork", "chickens?", "turkey", "lamb", "veal", "mutton",
    "venison", "fish", "ham", "bacon", "lard", "tallow", "shrimps?", "prawns?",
    "tuna", "salmon", "anchov(?:y|ies)", "sausages?", "crabs?", "lobsters?",
    "oysters?", "mussels?", "squid", "octopus", "clams?", "scallops?", "cod",
    "haddock", "mackerel", "herring", "sardines?", "trout", "ducks?", "goose",
    "bones?", "isinglass", "animal fats?", "animal rennet", "carmine", "cochineal",
    # Französisch
    "viandes?", "porc", "boeufs?", "veau", "agneau", "poulets?", "dinde", "canard",
    "oie", "jambon", "lardons?", "saucissons?", "saucisses?", "poissons?", "thon",
    "saumon", "anchois", "maquereau", "cabillaud", "morue", "crevettes?", "crabe",
    "homard", "moules?", "huitres?", "calmars?", "seiche", "poulpe", "foie",
    "boudin", "graisse animale", "gras de porc", "presure animale",
    # Spanisch
    "carnes?", "cerdo", "ternera", "vaca", "cordero", "pollo", "pavo", "pato", "jamon",
    "tocino", "panceta", "chorizo", "salchichas?", "salchichon", "mortadela",
    "pescados?", "atun", "sardinas?", "anchoas?", "boquerones", "bacalao", "merluza",
    "gambas?", "langostinos?", "cangrejo", "langosta", "mejillon(?:es)?", "ostras?",
    "calamar(?:es)?", "pulpo", "sepia", "higado", "sangre", "huesos?",
    "grasa animal", "cuajo animal", "manteca de cerdo",
    # Italienisch
    "maiale", "suino", "manzo", "bovino", "vitello", "agnello", "tacchino", "anatra",
    "prosciutto", "guanciale", "salame", "salsiccia", "mortadella", "pesce", "tonno",
    "salmone", "acciughe?", "alici", "merluzzo", "gamberi", "gamberetti", "gambero",
    "granchio", "aragosta", "cozze", "ostriche", "calamari", "polpo", "fegato",
    "sangue", "strutto", "grasso animale", "caglio animale",
    # Niederländisch
    "varken", "kalf", "kip", "kalkoen", "eend", "spek", "worst", "vis", "tonijn",
    "zalm", "ansjovis", "kabeljauw", "garnalen", "krab", "kreeft", "mosselen",
    "oesters", "inktvis", "bloed", "reuzel", "dierlijk vet",
    # Portugiesisch
    "porco", "boi", "vitela", "borrego", "cordeiro", "frango", "presunto", "toucinho",
    "linguica", "peixes?", "atum", "salmao", "sardinhas?", "anchovas?", "bacalhau",
    "camarao", "camaroes", "caranguejo", "lagosta", "mexilhao", "mexilhoes", "lulas?",
    "polvo", "figado", "banha", "ossos?", "gordura animal",
    # Polnisch
    "mies[oa]", r"miesn\w*", r"wieprzow\w*", r"wolow\w*", r"cielec\w*", r"jagnie\w*",
    "drob", r"kurczak\w*", r"indyk\w*", "kaczk[ai]", r"szynk\w*", "boczek",
    r"kielbas\w*", "ryba", "ryby", "ryb", "rybie", r"rybn\w*", r"tunczyk\w*", r"losos\w*",
    r"sledz\w*", r"sardynk\w*", r"dorsz\w*", r"krewetk\w*", r"krab\w*", r"homar\w*",
    "malze", r"ostryg\w*", r"kalmar\w*", r"zelatyn\w*", "tluszcz zwierzecy", r"watrob\w*",
    "krew",
    # Kroatisch / Tschechisch / Türkisch (Basis)
    "meso", "maso", "svinjetina", "govedina", "piletina", "riba", "ribe", "salama",
    "veprove", "hovezi", "tavuk", "sigir", "balik", "sucuk",
    # E-Nummern (Karmin, Gelatine, Knochenphosphat)
    "e120", "e441", "e542",
]

# Ausnahmen für die "nicht-vegetarisch"-Prüfung (werden vor der Suche aus dem Text entfernt)
EXC_NV = [
    r"butterschmalz", r"austernpilz\w*", r"austernseitling\w*", r"oyster mushrooms?",
    r"muschelnudel\w*", r"lamb'?s lettuce", r"crab apples?",
    r"(?:pre)?gelatini[sz]\w*",                      # "pregelatinised starch", "gelatinisierte Stärke"
    r"(?:vegetable|plant|vegetal\w*|pflanzlich\w*|vegan\w*|vegetarisch\w*|vegetarian\w*) gelatin\w*",
    # pflanzliches "Fleisch"/"Fruchtfleisch" etc.
    (r"(?:frucht|kokos|kokosnuss|tomaten|apfel|birnen|orangen|zitronen|limetten|kurbis|mango|"
     r"pflaumen|bananen|beeren|kirsch|pfirsich|aprikosen|ananas|melonen|avocado|kakao|nuss|"
     r"soja|sojabohnen|weizen|seitan|pflanzen|gemuse|erbsen|lupinen|jackfruit|pilz|champignon|"
     r"aubergine)fleisch\w*"),
    r"fleischtomat\w*", r"fleischlos\w*", r"fleischfrei\w*", r"fleischersatz\w*", r"fleischalternativ\w*",
    r"meat (?:alternatives?|substitutes?|analogu?e?s?|replacements?)",
]

# ---- NICHT VEGAN (aber vegetarisch): Milch, Ei, Honig, Bienen-/Insektenprodukte
NVEGAN_TEIL = [
    # Deutsch
    "milch", "sahne", "butter", "käse", "kaese", "joghurt", "jogurt", "yoghurt", "quark",
    "molke", "kasein", "casein", "laktose", "lactose", "lactosa", "lattosio", "laktoz",
    "laktalbumin", "lactalbumin", "lactoglobulin", "laktoglobulin", "lactoserum",
    "kazein", "schmand", "kefir", "eier", "eigelb", "eiklar", "eipulver", "vollei",
    "hühnerei", "honig", "bienenwachs", "gelée royale", "propolis", "schellack",
    "mozzarella", "parmesan", "gouda", "emmentaler", "cheddar", "mascarpone",
    "ricotta", "camembert", "gorgonzola", "pecorino", "halloumi",
    # Niederländisch
    "melk", "kaas", "slagroom", "bijenwas", "eigeel",
    # weitere Sprachen (mehrwortig)
    "cire d'abeille", "cera de abeja", "cera d'api", "cera de abelha", "jalea real",
    "geleia real", "pappa reale", "wosk pszczeli", "royal jelly",
]

NVEGAN_WORT = [
    # Deutsch
    "ei", "eiweiss", "rahm", "feta", "brie", "topfen", "obers", "ghee",
    # E-Nummern (Bienenwachs, Schellack, Lanolin, Lactit, Lysozym)
    "e901", "e904", "e913", "e966", "e1105",
    # Englisch
    "milks?", "creams?", "cheeses?", "yogurts?", "yoghurts?", "whey", "eggs?", "honey",
    "beeswax", "caseinates?", "dairy", "albumen", "shellac",
    # Französisch
    "lait", "laitiers?", "laitieres?", "beurre", "fromages?", "yaourts?", "yogourts?",
    "oeufs?", "miel", "babeurre", "fraiche",
    # Spanisch
    "leche", "lacteos?", "lacteas?", "mantequilla", "quesos?", r"yogur(?:es)?", "huevos?",
    "yemas?", "nata",
    # Italienisch
    "latte", "latticini", "burro", "formaggio", "formaggi", "uovo", "uova", "tuorlo",
    "albume", "miele", "panna",
    # Niederländisch
    "boter", "eieren", "honing",
    # Portugiesisch
    "leite", "manteiga", "queijos?", "iogurtes?", "ovos?", "gemas?",
    # Polnisch / Kroatisch / Tschechisch / Türkisch (Basis)
    "jajo", "jaja", "jajka", "jaj", r"jajeczn\w*", r"mlek\w*", r"mleczn\w*", "maslo",
    "sery", "sera", "serem", "ser", r"serow\w*", r"jogurt\w*", r"smietan\w*", r"serwatk\w*",
    "miod", r"zoltk\w*", "mlijeko", "jaje", "maslac", "sir", "vejce", "syr", "smetana",
    "vrhnje", "sut", "yumurta", "peynir", "tereyagi",
]

# Pflanzliche Alternativen: "Kokosmilch", "lait de coco", "mleko kokosowe", "Erdnussbutter" ...
_PFLANZE = (
    r"soj\w*|soy\w*|soia|avoine|oats?|hafer\w*|avena|haver|owsian\w*|amande\w*|almond\w*|"
    r"mandel\w*|almendra\w*|mandorla\w*|amandel\w*|migdal\w*|kokos\w*|coco|coconut|cocco|"
    r"riz|reis\w*|rice|arroz|riso|rijst|ryz\w*|noisette\w*|hazelnuts?|haselnuss\w*|avellana\w*|"
    r"nocciola\w*|hazelnoot|orzech\w*|cajou|cashew\w*|anacardo|erbse\w*|pois|peas?|guisante\w*|"
    r"pisello|erwt\w*|groch\w*|lupin\w*|dinkel\w*|spelt|epeautre|hanf\w*|hemp|chanvre|"
    r"nuss\w*|nuts?|erdnuss\w*|peanut\w*|pinda\w*|cacahuet\w*|arachide\w*|amendoim|"
    r"pflanz\w*|plant|vegetal\w*|sesam\w*|sesame|lein\w*|flax"
)
_MILCHARTIG = (
    r"milch\w*|milk\w*|lait\w*|leche\w*|latte|leite|melk\w*|mleko\w*|mlek\w*|mlijeko|"
    r"sahne\w*|cream\w*|creme\w*|nata|panna|smietan\w*|jogurt\w*|yogurt\w*|yoghurt\w*|"
    r"yaourt\w*|yogur\w*|iogurte\w*|kase\w*|kaas|cheese\w*|fromage\w*|queso\w*|formagg\w*|"
    r"drink|boisson|bebida|bevanda|butter\w*|beurre|mantequilla|burro|boter|maslo|manteiga"
)
_BUTTERPFLANZE = (
    r"cacao|kakao\w*|cocoa|cacau|karite|shea|illipe|mango|mangue|kokos\w*|coco|coconut|cocco|"
    r"erdnuss\w*|peanut|pinda|cacahuete|arachide|amendoim|nuss\w*|nut|mandel\w*|almond|amande|"
    r"haselnuss\w*|hazelnut|noisette|sesam\w*|sesame|sonnenblumen\w*|sunflower|tournesol|"
    r"girasol|cashew\w*|cajou|orzech\w*"
)
_BUTTERART = r"butter\w*|beurre|mantequilla|manteca|burro|boter|maslo|manteiga|pindakaas|kaas"
_MILCH_ROM = (
    r"lait|leche|latte|leite|creme|crema|nata|panna|beurre|mantequilla|manteca|burro|"
    r"manteiga|mleko|maslo|mlijeko|melk|boter|yaourt|yogur|fromage|queso|formaggio"
)
_VERB = r"(?:de|di|d'|du|of|van|z)?"

EXC_NVEGAN = [
    rf"\b(?:{_PFLANZE})[\s\-]*(?:{_MILCHARTIG})\b",                                # Kokosmilch, oat drink
    rf"\b(?:{_BUTTERPFLANZE})[\s\-]*(?:{_BUTTERART})\b",                           # Kakaobutter, peanut butter
    rf"\b(?:{_BUTTERART})[\s\-]*{_VERB}[\s\-]*(?:{_BUTTERPFLANZE})\w*\b",          # beurre de cacao
    rf"\b(?:{_MILCH_ROM})[\s\-]*{_VERB}[\s\-]*(?:{_PFLANZE})\w*\b",                # lait de coco, mleko kokosowe
    r"\b(?:vegan\w*|plant[\s\-]based|pflanzlich\w*|vegetal\w*)[\s\-]*"
    r"(?:milch|milk|kase|cheese|butter|sahne|cream|joghurt|yogurt|yoghurt|eier?|eggs?)\w*",
    r"(?:milch|kase|sahne|butter|joghurt|eier|ei)[\s\-]?(?:ersatz|alternative)\w*",  # Ei-Ersatz
    r"egg[\s\-]?(?:replacer|substitute|alternative)s?",
    r"milchsaure\w*", r"kwas mlekow\w*", r"melkzuur",                               # Milchsäure
    r"honigmelon\w*", r"butter ?nuts?", r"butter beans?", r"butterbohn\w*", r"butterfly",
    r"egg ?plants?", r"nata de coco", r"cream of tartar", r"pindakaas",
    r"pflanzlich\w* eiweiss",
]

# Spuren-/Herstellungshinweise: alles ab hier wird ignoriert (mehrsprachig)
_SPUREN = re.compile("|".join([
    r"kann spuren", r"spuren von", r"spuren enthalten", r"enthalt spuren",
    r"kann [^.;:]{0,60}enthalten", r"hergestellt in einem", r"hergestellt in einer",
    r"may contain", r"traces? of", r"produced in a", r"processed in a", r"manufactured in a",
    r"peut contenir", r"traces? de", r"fabrique dans (?:un|une)", r"produit dans (?:un|une)",
    r"puede contener", r"trazas? de", r"elaborado en (?:un|una)",
    r"puo contenere", r"tracce di", r"prodotto in (?:uno|un)",
    r"kan bevatten", r"kan sporen", r"sporen van", r"geproduceerd in",
    r"moze zawierac", r"slady", r"pode conter", r"vestigios de",
]))

# Negationen: "milchfrei", "meat-free", "ohne Milch", "sans lait" ...
_NEGATION = re.compile(
    r"\b\w+-?(?:frei|freie|freier|freien|freies|free)\b"
    r"|\b(?:meat|dairy|milk|lactose|egg|animal|gelatin)\s+free\b"
    r"|\b(?:ohne|without|sans|sin|senza|zonder|bez|sem)\s+\w+"
)

# Grundvokabular zur Spracherkennung (wenn nichts Tierisches gefunden wurde)
_VOKABULAR = set("""
wasser zucker salz mehl weizenmehl weizen mais reis hafer roggen dinkel starke maisstarke
kartoffelstarke weizenstarke sonnenblumenol rapsol palmol palmfett pflanzenol pflanzenfett
kokosfett zitronensaure aroma aromen naturliche naturliches gewurze gewurz hefe backpulver
tomaten tomatenmark kartoffeln zwiebeln knoblauch kakao kakaomasse vitamine vitamin
konservierungsstoff emulgator farbstoff antioxidationsmittel verdickungsmittel
suessungsmittel sirup glukosesirup fruktose dextrose maltodextrin pfeffer essig senf soja
linsen bohnen kichererbsen nusse mandeln haselnusse erdnusse rosinen apfel orangen zitronen
und aus mit von enthalt
water sugar salt flour wheat corn maize rice oats oat rye starch cornstarch sunflower palm
rapeseed oil fat vegetable acid citric flavouring flavoring flavourings flavors natural
spices spice yeast raising agent tomato tomatoes potato potatoes onion garlic cocoa vitamins
preservative emulsifier colour color antioxidant thickener sweetener syrup glucose fructose
pepper vinegar mustard soy soya lentils beans chickpeas nuts almonds hazelnuts peanuts
raisins apple orange lemon and of with from contains
eau sucre sel farine ble avoine seigle amidon huile tournesol colza vegetale acide citrique
arome aromes naturel naturels epices levure tomate pomme terre pommes oignon ail vitamine
conservateur emulsifiant colorant antioxydant epaississant edulcorant sirop poivre vinaigre
moutarde lentilles haricots pois chiches noix amandes noisettes cacahuetes raisins citron
et de du des avec contient
agua azucar sal harina trigo maiz arroz avena centeno almidon aceite girasol palma vegetal
acido citrico natural naturales especias levadura patata cebolla ajo vitamina vitaminas
conservante emulgente colorante antioxidante espesante edulcorante jarabe glucosa fructosa
pimienta vinagre mostaza lentejas judias garbanzos nueces almendras avellanas pasas manzana
naranja limon con del los las contiene
acqua zucchero sale farina grano riso segale amido olio girasole palma vegetale acido
aromi naturale naturali spezie lievito pomodoro patate cipolla aglio emulsionante colorante
antiossidante addensante dolcificante sciroppo glucosio fruttosio pepe aceto senape soia
lenticchie fagioli ceci noci mandorle nocciole arachidi uvetta mela arancia limone di
contiene
suiker zout bloem meel tarwe rijst haver rogge zetmeel olie zonnebloem raapzaad plantaardig
citroenzuur smaakstof kruiden gist aardappel ui knoflook conserveermiddel kleurstof
verdikkingsmiddel zoetstof siroop peper azijn mosterd linzen bonen kikkererwten noten
amandelen hazelnoten pinda rozijnen sinaasappel citroen van bevat
woda cukier sol maka pszenna pszenica kukurydza ryz owies zyto skrobia olej slonecznikowy
rzepakowy palmowy roslinny kwas cytrynowy aromat aromaty naturalne przyprawy drozdze
pomidor pomidory ziemniaki cebula czosnek witamina witaminy konserwant barwnik zageszczacz
slodzik glukoza fruktoza pieprz ocet musztarda soczewica fasola ciecierzyca orzechy migdaly
laskowe orzeszki ziemne rodzynki jablko pomarancza cytryna oraz zawiera
acucar farinha milho aveia centeio amido oleo girassol colza acido citrico especiarias
fermento batata cebola alho cacau conservante emulsionante corante espessante adocante
xarope frutose pimenta vinagre mostarda lentilhas feijao grao nozes amendoas avelas
amendoim passas maca laranja limao contem
""".split())


# =============================================================================
# 3. Regex kompilieren
# =============================================================================
# ---- Ergänzungen (Fälle aus echten OFF-Daten: Putenbrust, Entenbrust, Jambonneau, Insekten ...) ----
NV_TEIL += ["jambon", "kjott"]                       # Jambonneau (fr), Oksekjøtt (no)
NV_WORT += [
    # deutsche Tiernamen als Wortanfang: Putenbrust, Entenbrustfilet, Gänsebrust, Rinderhack ...
    r"rinder\w*", r"rinds\w*", r"kalb\w*", r"lamm\w*", r"huhn\w*", r"schwein\w*", r"pute\w*",
    r"enten\w*", r"ganse\w*", r"leber\w*", r"kaninchen\w*", r"faschiert\w*",
    "mett", "jerky", "biltong", "gans",
    # Insekten (zählen nicht als vegetarisch)
    "grillen", "grille", "heimchen", r"mehlwurm\w*", r"insekt\w*", r"larve\w*", r"heuschrecke\w*",
    r"buffalowurm\w*", "crickets?", "mealworms?", "locusts?", "grasshoppers?", "insects?",
    "larvae", "tenebrio", "acheta", "locusta", "gryllodes", "criquets?", "grillons?", "insectes?",
    # Skandinavisch
    "okse", "svin", r"kylling\w*", r"kyckling\w*", "skinke", "skinka", "laks", "torsk",
    r"fisk\w*", "reker", "rejer", r"polse\w*", r"korv\w*", "flesk",
    "couenne",
]
EXC_NV += [r"huhnerei\w*", r"zum grillen", r"ganseblum\w*"]
NVEGAN_TEIL += ["maelk", "mjolk", "honning", "honung"]
NVEGAN_WORT += ["ost", "smor", "aeg", "flode", "gradde"]

_VOKABULAR |= set("""
paprika paprikapulver koriander kurkuma curry zimt vanille ingwer chili kummel oregano
basilikum petersilie rosmarin thymian muskat lauch karotten mohren gurken spinat sellerie
kurbis zwiebel olivenol meersalz speisesalz jodiertes jodiertem citronensaure rapsol
coriander turmeric cinnamon vanilla ginger chilli cumin basil parsley rosemary thyme nutmeg
leek carrots cucumber spinach celery pumpkin olive iodised iodized
coriandre curcuma cannelle gingembre piment basilic persil romarin thym muscade poireau
carottes concombre epinards celeri citrouille
canela jengibre comino albahaca perejil romero tomillo puerro zanahoria pepino espinacas apio
cannella zenzero cumino basilico prezzemolo rosmarino timo porro carote cetriolo spinaci sedano
kaneel gember komijn basilicum peterselie rozemarijn tijm prei wortel komkommer spinazie
cynamon imbir kminek bazylia pietruszka rozmaryn tymianek marchew ogorek szpinak seler dynia
gengibre cominho manjericao alecrim tomilho cenoura espinafre aipo abobora
""".split())

# Kontext (Produktname/Kategorie): Hinweise auf Fleisch- oder Fischprodukte ...
_ERSATZ_RE = re.compile(
    r"vegan|veggie|vegetar|plant|pflanz|ersatz|alternativ|analog|substitut|imitation|soja|\bsoy|"
    r"seitan|tofu|tempeh|jackfruit|lupin|vegetal|\btyp\b|nach art|\bfree\b|\bfrei\b"
)
# ... die bei Gewürzen, Aromen und "Geschmack"-Produkten nichts über die Zutaten sagen
_AROMA_RE = re.compile(r"geschmack|flavou?r|gewurz|wurzmischung|seasoning|spice|\bgout\b|sabor|sapore")


# ---- Ergänzungen 2: Käse, Milchprodukte, Fisch (inkl. Tippfehler aus den OFF-Daten) ----
NV_TEIL += ["seehecht", "sprott", "wolfsbarsch", "octopus", "oktopus", "markrele", "hareng",
            "thunfisch", "seelachs", "pangasius", "dorsch", "karpfen", "scholle", "zander",
            "sprat", "sardin"]
NV_WORT += [r"fish(?:es)?", r"seafoods?", r"meats?", "poultry", r"mackerels?", r"herrings?",
            r"octopusse", r"harengs?"]
NVEGAN_TEIL += ["frischkase", "frischkass", "scamorza", "sauerrahm", "schlagrahm", "schlagsahne",
                "ghurt", "ghourt", "johgurt", "j0ghurt", "jogh", "stremsel", "presure",
                "caglio", "cuajo"]
NVEGAN_WORT += [r"rahm\w*", "smetana", r"podpuszczk\w*"]

# Algen/Seetang sind pflanzlich - auch wenn die Kategorie "Seafood" heißt
_ALGEN_RE = re.compile(r"\balg(?:e|en|a|as|ue|ues)\b|seealg|seetang|seaweed|\bnori\b|kelp|wakame|"
                       r"\bagar|purpurtang|\bdulse\b|\btang\b|spirulina|chlorella")
# Pflanzliche Zutaten im Produktnamen/der Kategorie: "Kokos Natur (Yoghurt)" ist kein Milchprodukt
_PFLANZE_KTX = re.compile(r"kokos|coconut|\bcoco\b|hafer|\boats?\b|\boat\b|mandel|almond|soja|soya|\bsoy|"
                          r"cashew|lupin|pflanz|plant|vegan|veggie|erbsen|\bpea\b|dairy[\s\-]?free|"
                          r"milchfrei|lactose[\s\-]?free")


def _kompiliere(teil: list[str], woerter: list[str]) -> re.Pattern:
    teile = sorted({re.escape(_ascii(t)) for t in teil}, key=len, reverse=True)
    worte = sorted({_ascii(w) for w in woerter}, key=len, reverse=True)
    return re.compile("(?:" + "|".join(teile) + r")|\b(?:" + "|".join(worte) + r")\b")


def _kompiliere_ausnahmen(muster: list[str]) -> re.Pattern:
    return re.compile("|".join(_ascii(m) for m in muster))


_NV_RE = _kompiliere(NV_TEIL, NV_WORT)
_NVEGAN_RE = _kompiliere(NVEGAN_TEIL, NVEGAN_WORT)
_EXC_NV_RE = _kompiliere_ausnahmen(EXC_NV)
_EXC_NVEGAN_RE = _kompiliere_ausnahmen(EXC_NVEGAN)


# =============================================================================
# 4. Analyse des Zutatentexts
# =============================================================================
def _analysiere_text(text) -> tuple:
    """Gibt (level, treffer, bekannt) zurück.

    level: 0 = Fleisch/Fisch gefunden, 1 = Milch/Ei/Honig gefunden,
           2 = nichts Tierisches UND Sprache erkannt, None = Text unbrauchbar
    """
    if not isinstance(text, str):
        return None, (), False
    t = normalisiere(text)
    if len(t) < 3:
        return None, (), False
    # Nicht-lateinische Schrift (kyrillisch, griechisch, arabisch ...) -> Text unbrauchbar
    buchstaben = [c for c in text if c.isalpha()]
    if buchstaben and sum(ord(c) > 0x024F for c in buchstaben) / len(buchstaben) > 0.1:
        return None, (), False

    m = _SPUREN.search(t)                 # Spurenhinweise abschneiden
    if m:
        t = t[:m.start()]
    t = _NEGATION.sub(" ", t)             # "milchfrei", "ohne Ei" ... entfernen

    nv = sorted(set(_NV_RE.findall(_EXC_NV_RE.sub(" ", t))))
    if nv:
        return 0, tuple(nv), True

    nvegan = sorted(set(_NVEGAN_RE.findall(_EXC_NVEGAN_RE.sub(" ", t))))
    if nvegan:
        return 1, tuple(nvegan), True

    verschieden = set(re.findall(r"[a-z]{2,}", t))
    treffer = len(verschieden & _VOKABULAR)
    # Sprache gilt als erkannt, wenn genug Grundwörter vorkommen UND sie einen
    # nennenswerten Anteil des Textes ausmachen (verhindert Zufallstreffer wie "sol")
    bekannt = treffer >= 3 or (treffer >= 1 and treffer / max(len(verschieden), 1) >= 0.3)
    return (2 if bekannt else None), (), bekannt


# =============================================================================
# 5. Analyse der OFF-Tags
# =============================================================================
def _tag_status(tags) -> tuple[str, str]:
    """(vegetarisch-Status, vegan-Status) mit Werten ja/nein/maybe/unbekannt/fehlt."""
    if isinstance(tags, (list, tuple, set)):
        tags = ",".join(map(str, tags))
    if not isinstance(tags, str) or not tags.strip():
        return "fehlt", "fehlt"
    s = {t.strip().lower() for t in tags.split(",")}

    def status(ja, nein, vielleicht, unbekannt):
        if nein in s:
            return "nein"
        if ja in s:
            return "ja"
        if vielleicht in s:
            return "maybe"
        if unbekannt in s:
            return "unbekannt"
        return "fehlt"

    veg = status("en:vegetarian", "en:non-vegetarian", "en:maybe-vegetarian", "en:vegetarian-status-unknown")
    vegan = status("en:vegan", "en:non-vegan", "en:maybe-vegan", "en:vegan-status-unknown")
    return veg, vegan


# =============================================================================
# 6. Kombination beider Quellen
# =============================================================================
def _kontext_treffer(kontext, text=None) -> tuple:
    """Hinweise aus Produktname + Kategorie: (level, treffer). level 0 = Fleisch/Fisch, 1 = Milch/Ei.

    Gibt (None, ()) zurück, wenn Name/Kategorie ein Ersatzprodukt, eine Alge oder ein
    Gewürz/Aroma nahelegen.
    """
    name, kategorie, marke = kontext
    ctx = _ascii(f"{name or ''} | {kategorie or ''}")
    alles = f"{ctx} | {_ascii(marke or '')}"
    if _ERSATZ_RE.search(alles) or _AROMA_RE.search(ctx):
        return None, ()
    if _ALGEN_RE.search(alles) or (isinstance(text, str) and _ALGEN_RE.search(_ascii(text))):
        return None, ()
    nv = tuple(sorted(set(_NV_RE.findall(_EXC_NV_RE.sub(" ", ctx)))))
    if nv:
        return 0, nv
    if _PFLANZE_KTX.search(alles):
        return None, ()
    nvegan = tuple(sorted(set(_NVEGAN_RE.findall(_EXC_NVEGAN_RE.sub(" ", ctx)))))
    if nvegan:
        return 1, nvegan
    return None, ()


@lru_cache(maxsize=300_000)
def _klassifiziere(text, tags, kontext=None) -> tuple:
    level, treffer, bekannt = _analysiere_text(text)
    veg, vegan = _tag_status(tags)

    text_treffer = level if level in (0, 1) else None     # Treffer = tierische Zutat
    text_sauber = level == 2

    # Kontext (Name/Kategorie) darf nur eingreifen, wenn Text UND Tags nicht gemeinsam "vegan" sagen
    # Geschützt: Tags sagen "vegan" UND der Zutatentext ist sauber (oder ein langer Text ohne Tierisches)
    geschuetzt = vegan == "ja" and (text_sauber or (
        isinstance(text, str) and len(text) >= 25 and level not in (0, 1)))
    ktx_level, ktx_treffer = (None, ())
    if kontext and not geschuetzt:
        ktx_level, ktx_treffer = _kontext_treffer(kontext, text)

    # Obergrenze, die jede Quelle zulässt (0 = nicht-veg, 1 = höchstens vegetarisch, 2 = offen)
    tag_cap = 0 if veg == "nein" else 1 if vegan == "nein" else 2
    tag_floor = 2 if vegan == "ja" else 1 if veg == "ja" else 0
    cap = min(tag_cap,
              text_treffer if text_treffer is not None else 2,
              ktx_level if ktx_level is not None else 2)

    if cap == 0:
        label = NICHT_VEG
    elif cap == 1:
        label = VEGETARISCH
    elif vegan == "ja" or text_sauber:
        label = VEGAN
    else:
        label = UNBEKANNT

    text_est = level                                       # 0/1/2 oder None
    widerspruch = text_est is not None and (text_est > tag_cap or text_est < tag_floor)
    if ktx_level == 0 and (text_sauber or vegan == "ja" or veg == "ja"):
        widerspruch = True                                 # Kontext sagt Fleisch, Text/Tags sagen sauber
    if ktx_level == 1 and (text_sauber or vegan == "ja"):
        widerspruch = True                                 # Kontext sagt Milch/Ei, Text/Tags sagen vegan

    # Quelle
    if label == UNBEKANNT:
        quelle = "keine"
    else:
        quellen = []
        if (text_treffer == cap) if cap < 2 else text_sauber:
            quellen.append("Text")
        if (tag_cap == cap) if cap < 2 else (vegan == "ja"):
            quellen.append("Tags")
        if ktx_level is not None and cap == ktx_level:
            quellen.append("Kontext")
        quelle = "+".join(quellen) or "Tags"

    # Konfidenz
    if label == UNBEKANNT:
        konfidenz = "keine"
    elif widerspruch:
        konfidenz = "niedrig"
    elif label == VEGAN:
        konfidenz = "hoch" if (vegan == "ja" and text_sauber) else "mittel"
    elif text_treffer == cap:
        konfidenz = "hoch"
    elif label == VEGETARISCH and veg != "ja":
        konfidenz = "niedrig"            # "nicht vegan", aber unklar ob Fleisch enthalten ist
    else:
        konfidenz = "mittel"

    # Details zum Nachvollziehen
    teile = []
    if treffer:
        teile.append("Text: " + ", ".join(treffer[:8]))
    elif text_sauber:
        teile.append("Text: nichts Tierisches")
    elif isinstance(text, str):
        teile.append("Text: zu kurz oder Sprache nicht erkannt")
    if ktx_treffer:
        teile.append("Kontext: " + ", ".join(ktx_treffer[:5]))
    if veg != "fehlt" or vegan != "fehlt":
        teile.append(f"Tags: vegetarisch={veg}, vegan={vegan}")
    return label, quelle, konfidenz, bool(widerspruch), " | ".join(teile)


def _bereinige(wert):
    if wert is None or (isinstance(wert, float) and pd.isna(wert)):
        return None
    if isinstance(wert, (list, tuple, set)):
        return ",".join(map(str, wert))
    return wert if isinstance(wert, str) else str(wert)


def klassifiziere(text, tags=None, name=None, kategorie=None, marke=None) -> dict:
    """Einzelnes Produkt klassifizieren (Zutatentext, OFF-Tags, optional Name/Kategorie/Marke)."""
    kontext = (_bereinige(name), _bereinige(kategorie), _bereinige(marke))
    if not any(kontext):
        kontext = None
    label, quelle, konfidenz, widerspruch, details = _klassifiziere(
        _bereinige(text), _bereinige(tags), kontext)
    return {"label": label, "quelle": quelle, "konfidenz": konfidenz,
            "widerspruch": widerspruch, "details": details}


def klassifiziere_dataframe(df: pd.DataFrame, text_col: str = "ingredients_text",
                            tags_col: str = "ingredients_analysis_tags",
                            name_col: str = "product_name",
                            kategorie_col: str = "main_category_en",
                            marke_col: str = "brands",
                            praefix: str = "diet_type") -> pd.DataFrame:
    """Klassifiziert alle Zeilen. Gibt ein DataFrame mit fünf neuen Spalten zurück.

    Name-, Kategorie- und Markenspalte sind optional (fehlen sie im DataFrame, werden sie ignoriert).
    """
    def spalte(name):
        return df[name] if name in df.columns else [None] * len(df)

    ergebnis = []
    for t, g, n, k, m in zip(df[text_col], df[tags_col], spalte(name_col),
                             spalte(kategorie_col), spalte(marke_col)):
        kontext = (_bereinige(n), _bereinige(k), _bereinige(m))
        ergebnis.append(_klassifiziere(_bereinige(t), _bereinige(g), kontext if any(kontext) else None))
    return pd.DataFrame(ergebnis, index=df.index, columns=[
        praefix, f"{praefix}_quelle", f"{praefix}_konfidenz",
        f"{praefix}_widerspruch", f"{praefix}_details"])


def ergaenze_diet_type(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Hängt diet_type + Zusatzspalten an df an und gibt das neue DataFrame zurück.

    Kann beliebig oft ausgeführt werden: alte diet_type*-Spalten werden vorher entfernt,
    es gibt also nie einen Fehler wegen doppelter Spaltennamen.
        df = ergaenze_diet_type(df)
    """
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("diet_type")])
    return df.join(klassifiziere_dataframe(df, **kwargs))


def haeufige_woerter(serie: pd.Series, n: int = 60, min_laenge: int = 3) -> pd.Series:
    """Hilfsmittel zum Verbessern der Listen: häufigste Wörter in einer Auswahl von Texten.

    Beispiel: haeufige_woerter(df.loc[df.diet_type_widerspruch, "ingredients_text"])
    """
    zaehler = Counter()
    for t in serie.dropna():
        zaehler.update(re.findall(rf"[a-z]{{{min_laenge},}}", normalisiere(t)))
    return pd.Series(zaehler).sort_values(ascending=False).head(n)


# ---- Kompatibilität mit den alten Funktionsnamen ----------------------------
def erkenne_ernaehrungsform_aus_zutatenliste(text) -> str:
    return klassifiziere(text, None)["label"]


def ernaehrungsform_aus_tags(tags) -> str:
    return klassifiziere(None, tags)["label"]


# =============================================================================
# 7. Selbsttest:  python diettype.py
# =============================================================================
_TESTS = [
    # (Text, Tags, erwartetes Label)
    ("weizenmehl, butterschmalz, salz", None, VEGETARISCH),
    ("wasser, kokosmilch, zucker", None, VEGAN),
    ("nudeln, austernpilz, salz", None, VEGAN),
    ("mehl, schweineschmalz", None, NICHT_VEG),
    ("reis, pigmente, wasser", None, VEGAN),
    ("mehl, eier, salz", None, VEGETARISCH),
    ("wasser, zucker, wachteleier", None, VEGETARISCH),
    # fremdsprachige Eier (der ursprüngliche Fehlerfall)
    ("Œufs de poules élevées en plein air", "en:palm-oil-free,en:non-vegan,en:vegetarian", VEGETARISCH),
    ("_œufs_", None, VEGETARISCH),
    ("Huevos de Gallinas Camperas", None, VEGETARISCH),
    ("jaja iz obogaćenih kaveza", "en:non-vegan,en:vegetarian", VEGETARISCH),
    ("97% pasteryzowane białko z kurzych jaj, 3% żółtko", None, VEGETARISCH),
    # Fallen bei Teilstrings
    ("Fruchtfleisch, Wasser, Zucker", None, VEGAN),
    ("Tomatenmark, Fleischtomaten, Salz", None, VEGAN),
    ("Weizenmehl, Wasser, pregelatinised starch", None, VEGAN),
    ("wild blueberries, sugar, water", None, VEGAN),
    ("lamb's lettuce, olive oil, salt", None, VEGAN),
    ("oyster mushrooms, water, salt", None, VEGAN),
    ("sugar, cocoa butter, cocoa mass", None, VEGAN),
    ("butterfly pea flower, water, sugar", None, VEGAN),
    ("pflanzliches eiweiß, wasser, salz", None, VEGAN),
    ("Haferdrink, Wasser, Salz", None, VEGAN),
    # pflanzliche Alternativen in mehreren Sprachen
    ("coconut milk (80%), water, sugar", None, VEGAN),
    ("lait de coco, eau, sucre", None, VEGAN),
    ("leche de almendras, agua, azúcar", None, VEGAN),
    ("latte di soia, acqua, zucchero", None, VEGAN),
    ("leite de coco, água, açúcar", None, VEGAN),
    ("mleko kokosowe, woda, cukier", None, VEGAN),
    ("kokosmelk, water, suiker", None, VEGAN),
    ("pindakaas, zout, olie", None, VEGAN),
    ("nata de coco, agua, azucar", None, VEGAN),
    ("Kokosmilch, Magermilchpulver, Zucker", None, VEGETARISCH),
    # tierische Zutaten in mehreren Sprachen
    ("lait écrémé, sucre, eau", None, VEGETARISCH),
    ("latte intero, zucchero", None, VEGETARISCH),
    ("volle melk, suiker", None, VEGETARISCH),
    ("zucker, kakaobutter, vollmilchpulver, emulgator", None, VEGETARISCH),
    ("Haferflocken, Honig, Salz", None, VEGETARISCH),
    ("Kartoffeln, Käse, Salz", None, VEGETARISCH),
    ("Hähnchenfleisch, Wasser, Salz", None, NICHT_VEG),
    ("pork, water, salt", None, NICHT_VEG),
    ("porc, eau, sel", None, NICHT_VEG),
    ("jamón, agua, sal", None, NICHT_VEG),
    ("zucker, gelatine, aroma", None, NICHT_VEG),
    ("Zucker, Wasser, Farbstoff: E 120", None, NICHT_VEG),
    # Spuren, Negationen
    ("Wasser, Zucker, Aroma. Kann Spuren von Milch und Eiern enthalten.", None, VEGAN),
    ("Wasser, Zucker, milchfreie Schokolade, Salz", None, VEGAN),
    # nur Tags
    (None, "en:vegan,en:vegetarian", VEGAN),
    (None, "en:non-vegan,en:vegetarian", VEGETARISCH),
    (None, "en:non-vegetarian,en:non-vegan", NICHT_VEG),
    (None, "en:vegan-status-unknown,en:vegetarian", UNBEKANNT),
    (None, None, UNBEKANNT),
    (float("nan"), float("nan"), UNBEKANNT),
    # Text unbrauchbar / Sprache unbekannt
    ("voda, šećer, sol, brašno", None, UNBEKANNT),
    ("voda, šećer, sol, brašno", "en:vegan,en:vegetarian", VEGAN),
    # Widersprüche: strengste Quelle gewinnt
    ("wasser, zucker, magermilchpulver", "en:vegan,en:vegetarian", VEGETARISCH),
    ("wasser, zucker, salz", "en:non-vegan,en:vegetarian", VEGETARISCH),
    # echte OFF-Fälle: Fleisch ohne erkannte Zutat im Text
    ("Putenbrustfilet, jodiertes Speisesalz", None, NICHT_VEG),
    ("97% flüssig gewürztes Putenbrustfilet", "en:vegan-status-unknown", NICHT_VEG),
    ("Entenbrustfilet mit Haut 92%, mariniert", None, NICHT_VEG),
    ("Gänsebrust (94%), jodiertes Nitritpöckelsalz", None, NICHT_VEG),
    ("Oksekjøtt, sukker, salt, tørket soya saus", None, NICHT_VEG),
    ("83% jambonneau sans couenne, 13% eau, sel", None, NICHT_VEG),
    ("Ganze getrocknete Mehlwürmer (Tenebrio molitor)", None, NICHT_VEG),
    ("Ganze getrocknete Grillen (Acheta domestica)", None, NICHT_VEG),
    ("Hühnerei, Wasser, Salz", None, VEGETARISCH),
    ("es dem aftig und der Grill) В Rostar ЗАМРАЗЕН", None, UNBEKANNT),
]


_KONTEXT_TESTS = [
    # (Text, Tags, Name, Kategorie, Marke, erwartet)
    (None, "en:vegan,en:vegetarian", "Bacon", "Bacon", None, NICHT_VEG),
    ("Salz,Rapsöl,E500,Gewürz,E331,Hefe,Weizenmehl", "en:vegan-status-unknown,en:vegetarian-status-unknown",
     "Hähnchen Schnitzel", "Breaded chicken", None, NICHT_VEG),
    ("Wasser, texturiertes Weizenprotein 16%, Rapsöl, Salz", "en:vegan,en:vegetarian",
     "Hauchzarter Aufschnitt Typ Hähnchen Klassik", "Prepared meat cuts substitutes", None, VEGAN),
    ("Wasser, 21% Weizeneiweiß, Paprikawürfel, Salz", "en:vegan,en:vegetarian",
     "Vegane cevapcici", "Meat analogues", None, VEGAN),
    (None, "en:vegan,en:vegetarian", "Bio-Faschiertes", "de:Veganes-faschiertes", "Spar, Spar Veggie", VEGAN),
    (None, "en:vegan,en:vegetarian", "Chicorée", "Belgian endives", None, VEGAN),
    ("Paprikapulver, Curry, Salz, Zucker", None, "Brathähnchen Gewürz", "en:Groceries", None, VEGAN),
    ("Wasser, Zucker, Salz", "en:vegan,en:vegetarian", "Mühlen Mett", None, None, VEGAN),
    # Käse / Milchprodukte
    ("Kerniger Frischkässe, jodiertes Speisesalz.", "en:vegan-status-unknown", "Kerniger Frischkäse", "Cream cheeses", None, VEGETARISCH),
    ("tait pasteurisé, sel. présure, correcteur d'acidité", None, "Scamorza", "Scamorza", None, VEGETARISCH),
    ("Zucker, Pfirsiche, Wasser, modifizierte Stärke", None, "Fruchtjoghurt", "Fruit yogurts", None, VEGETARISCH),
    ("_J0GHURT_ mild, Zucker, Pfirsiche, Wasser", None, "Almighurt", "Milchprodukte", None, VEGETARISCH),
    ("Mikrobielles Lab", "en:vegan,en:vegetarian", "Rahm Camembert 60% Fett", "Camemberts", None, VEGETARISCH),
    (None, "en:vegan-status-unknown", "Sprüh Sahne", "Whipped cream under pressure UHT", None, VEGETARISCH),
    ("SACE/TORINS 00 0g 0% 28.00 7.09 14% 1.00", "en:vegan-status-unknown", "Allgäuer Käse", "Sliced cheeses", None, VEGETARISCH),
    ("Wasser, Kokosmilch 21%, Zucker", "en:vegan-status-unknown", "Kokos Natur (Yoghurt)", "Yogurts", None, VEGAN),
    ("cashew nuts* (28,2%), almonds*, mango* (12%), salt, spices", "en:vegan,en:vegetarian", "Pa'lais Mango & Chili", "Cheeses", None, VEGAN),
    ("Wasser, Kokosfett, Stärke, Salz", "en:vegan,en:vegetarian", "Goudí", "Cheeses", None, VEGAN),
    ("Weizenmehl", "en:vegan,en:vegetarian", "Weizenmehl Bio", "Flours", None, VEGAN),
    # Fisch
    ("Salz, Maltodextrine, Traubenzucker, natürliches Aroma", "en:vegan-status-unknown", "Kap-Seehecht Zitrone & Pfeffer", "de:seehecht", None, NICHT_VEG),
    ("Sprotten 73%, Bio-Sonnenblumenöl 26%, Salz", None, "Kieler Sprotten", "Smoked fishes", None, NICHT_VEG),
    ("Markrele, Speisesalz, Edelholzrauch", None, "Markrele", "Smoked mackerels", None, NICHT_VEG),
    ("hareng avec peau, concombres, vinaigre", None, "Harengs roulés", "Rollmops", None, NICHT_VEG),
    ("Octopusse (60 %), Sonnenblumenöl, Salz", None, "Antipasti Di Mare", "Canned fishes", None, NICHT_VEG),
    ("44% Wolfsbarschfilet, 12% Tomaten, 10% Zwiebeln", None, "Gemüse mit Wolfsbarschfilet", "de:wolfsbarschfilets", None, NICHT_VEG),
    ("Seealgen (Undaria Pinnatifida) 78%, Füllstoff", None, "Salade de mer", "Seafood", None, UNBEKANNT),
    ("Seaweed (100%).", "en:vegan,en:vegetarian", "Sushi Nori", "Dried nori seaweed sheets", None, VEGAN),
    ("Trinkwasser, texturiertes Sojaprotein, Panade", "en:vegan,en:vegetarian", "Vantastic Fish Fingers", "de:vegetarische-fischstäbchen", None, VEGAN),
    ("Möhren 88 %, Wasser, Sonnenblumenöl", "en:vegan,en:vegetarian", "Räucher Lax", "de:veganer-lachs", None, VEGAN),
]


def _selbsttest() -> None:
    fehler = 0
    for text, tags, erwartet in _TESTS:
        ist = klassifiziere(text, tags)["label"]
        if ist != erwartet:
            fehler += 1
            print(f"FEHLER: {text!r} / {tags!r} -> {ist} (erwartet {erwartet})")
    for text, tags, name, kat, marke, erwartet in _KONTEXT_TESTS:
        ist = klassifiziere(text, tags, name, kat, marke)["label"]
        if ist != erwartet:
            fehler += 1
            print(f"FEHLER (Kontext): {name!r} / {text!r} -> {ist} (erwartet {erwartet})")
    # Widerspruch-Flag
    assert klassifiziere("wasser, zucker, magermilchpulver", "en:vegan,en:vegetarian")["widerspruch"]
    assert not klassifiziere("wasser, zucker, salz", "en:vegan,en:vegetarian")["widerspruch"]
    gesamt = len(_TESTS) + len(_KONTEXT_TESTS)
    print(f"{gesamt - fehler}/{gesamt} Tests bestanden")
    if fehler:
        raise SystemExit(1)


if __name__ == "__main__":
    _selbsttest()

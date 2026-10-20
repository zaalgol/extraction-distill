"""Scenario specs for synthetic documents: language, scenario type, stance, invented names.

The mix follows the gold set (claims and attacks, arrests, finance and wallets, online
infrastructure, meetings, denials, speculation, ~15% off-topic), but every name is sampled
here, so no gold entity can leak in through the generator's prompt. Person names are random
first/last combinations from regional pools; groups and organizations are random
adjective-noun-word combinations given in English and translated by the writer; towns are
mostly invented. Identifiers (handles, domains, wallets, phones, emails) are random strings.
"""

from __future__ import annotations

import random
import string
from dataclasses import asdict, dataclass, field

LANGS = {"en": "English", "ar": "Arabic", "ru": "Russian", "fr": "French", "es": "Spanish",
         "he": "Hebrew", "fa": "Persian (Farsi)", "tr": "Turkish", "de": "German"}
# English is half the gold set; the student must also be good in the others, so they get
# more weight here than in the gold set.
LANG_WEIGHTS = {"en": 0.28, "ar": 0.10, "ru": 0.09, "fr": 0.09, "es": 0.09, "he": 0.08,
                "fa": 0.08, "tr": 0.09, "de": 0.10}

# scenario -> weight. near_miss + irrelevant ~15%, like the gold set's off-topic share.
SCENARIOS = {
    "claim_attack": 0.11, "attack_reported": 0.06, "arrest": 0.11, "finance": 0.10,
    "infrastructure": 0.10, "meeting": 0.07, "leadership": 0.05, "affiliation": 0.05,
    "threat": 0.05, "denial": 0.07, "speculative": 0.07, "mixed_stance": 0.04,
    "near_miss": 0.10, "irrelevant": 0.05,
}
OFF_TOPIC = {"near_miss", "irrelevant"}

LENGTHS = {"1-3 sentences": 0.55, "4-7 sentences": 0.33, "2-4 short paragraphs": 0.12}

STYLES = ["a wire-service news report", "a local newspaper article", "a government press release",
          "a court report", "an NGO or think-tank brief", "a summary of a Telegram channel post",
          "a security analyst's blog post", "a radio news bulletin transcript",
          "a police statement"]

FIRST = {
    "arab": ["Omar", "Khalid", "Ahmad", "Hassan", "Tariq", "Bilal", "Mahmoud", "Nabil", "Rami",
             "Ziad", "Fadi", "Walid", "Samira", "Nour", "Huda", "Mustafa", "Jamal", "Anas",
             "Hamza", "Ibrahim", "Faisal", "Majid", "Rania", "Salma", "Adnan", "Bashir"],
    "russian": ["Dmitry", "Sergei", "Alexei", "Nikolai", "Pavel", "Oleg", "Viktor", "Andrei",
                "Mikhail", "Roman", "Yuri", "Anna", "Elena", "Irina", "Konstantin", "Ruslan",
                "Stanislav", "Denis", "Vadim", "Gleb"],
    "turkish": ["Mehmet", "Emre", "Burak", "Kerem", "Murat", "Serkan", "Hakan", "Ayşe", "Elif",
                "Volkan", "Cem", "Onur", "Tolga", "Deniz", "Zeynep", "Uğur", "Baran"],
    "persian": ["Reza", "Mehdi", "Ali", "Hossein", "Farhad", "Kaveh", "Babak", "Dariush",
                "Navid", "Shirin", "Parisa", "Arash", "Kian", "Omid", "Saeed", "Behnam"],
    "hebrew": ["Avi", "Eitan", "Noam", "Ronen", "Itai", "Yossi", "Gil", "Tamar", "Michal",
               "Oren", "Ariel", "Shira", "Lior", "Doron", "Erez", "Nadav"],
    "french": ["Julien", "Mathieu", "Nicolas", "Antoine", "Pierre", "Hugo", "Camille", "Claire",
               "Sophie", "Thomas", "Lucas", "Mehdi", "Yanis", "Karim", "Inès", "Rachid"],
    "spanish": ["Javier", "Diego", "Alejandro", "Carlos", "Mateo", "Lucía", "Carmen", "Pablo",
                "Andrés", "Sergio", "Rodrigo", "Valeria", "Hernán", "Esteban", "Joaquín"],
    "german": ["Lukas", "Jonas", "Felix", "Tobias", "Stefan", "Matthias", "Katrin", "Julia",
               "Florian", "Markus", "Sebastian", "Lena", "Dominik", "Jörg", "Uwe"],
    "english": ["James", "Connor", "Ethan", "Liam", "Owen", "Grace", "Hannah", "Marcus",
                "Victor", "Isaac", "Amara", "Kwame", "Tunde", "Chidi", "Nadia", "Bradley"],
}
LAST = {
    "arab": ["al-Masri", "Qassem", "Nasser", "Haddadin", "al-Tamimi", "Saleh", "Barakat",
             "al-Zubaidi", "Khoury", "Darwish", "Mansour", "al-Rifai", "Jaber", "Sabbagh",
             "al-Hamdani", "Kanaan", "Yassin", "al-Obeidi", "Zaher", "Awad"],
    "russian": ["Sokolov", "Morozov", "Lebedev", "Kozlov", "Novikov", "Orlov", "Zaitsev",
                "Pavlov", "Fedorov", "Gusev", "Tarasov", "Belyaev", "Komarov", "Vinogradov",
                "Kiselev", "Zhukov", "Nikitin", "Gromov"],
    "turkish": ["Yılmaz", "Demir", "Şahin", "Çelik", "Aydın", "Öztürk", "Arslan", "Doğan",
                "Kılıç", "Aslan", "Polat", "Korkmaz", "Erdem", "Kurt", "Özdemir", "Tekin"],
    "persian": ["Rahimi", "Karimi", "Moradi", "Hosseini", "Jafari", "Ahmadi", "Rostami",
                "Kazemi", "Sadeghi", "Bagheri", "Mohammadi", "Tehrani", "Shirazi", "Nouri"],
    "hebrew": ["Levi", "Mizrahi", "Peretz", "Biton", "Avraham", "Friedman", "Shapiro",
               "Ben-David", "Azoulay", "Katz", "Golan", "Segal", "Ohana", "Dahan"],
    "french": ["Lefebvre", "Moreau", "Girard", "Fournier", "Mercier", "Bonnet", "Lambert",
               "Rousseau", "Chevalier", "Garnier", "Faure", "Brahimi", "Bensaïd", "Morel"],
    "spanish": ["Ortega", "Navarro", "Romero", "Herrera", "Castillo", "Vargas", "Delgado",
                "Mendoza", "Iglesias", "Ríos", "Cabrera", "Fuentes", "Paredes", "Salazar"],
    "german": ["Schneider", "Fischer", "Weber", "Becker", "Hoffmann", "Wagner", "Krüger",
               "Schäfer", "Brandt", "Vogel", "Lorenz", "Hartmann", "Zimmermann", "Kaiser"],
    "english": ["Whitaker", "Mensah", "Adeyemi", "Callahan", "Thornton", "Okafor", "Brennan",
                "Holloway", "Asante", "Fletcher", "Danso", "Kimani", "Harlow", "Pryce"],
}
# Which name cultures are plausible in a document of each language.
CULTURES = {"en": ["english", "arab", "russian", "persian", "turkish", "french"],
            "ar": ["arab"], "ru": ["russian", "arab"], "fr": ["french", "arab"],
            "es": ["spanish", "arab"], "he": ["hebrew", "arab"], "fa": ["persian", "arab"],
            "tr": ["turkish", "arab", "persian"], "de": ["german", "turkish", "arab"]}

GROUP_ADJ = ["Silver", "Scarlet", "Hidden", "Eastern", "Western", "Southern", "Free", "Eternal",
             "Burning", "Golden", "Grey", "Green", "White", "Steel", "Stone", "Faithful",
             "Free Mountain", "Sacred", "United", "Revolutionary", "Popular", "Last", "First",
             "Desert", "Coastal", "Highland", "Red", "Blue"]
GROUP_NOUN = ["River", "Flame", "Sword", "Lion", "Falcon", "Banner", "Storm", "Eagle", "Wolf",
              "Star", "Horizon", "Gate", "Mountain", "Thunder", "Spear", "Wind", "Arrow",
              "Torch", "Oath", "Harvest", "Valley", "Fortress", "Lantern", "Phoenix"]
GROUP_WORD = ["Front", "Brigades", "Battalion", "Movement", "Army", "Legion", "Guard",
              "Partisans", "Vanguard", "Corps", "Faction", "Squadrons", "Resistance"]
ORG_KIND = ["Foundation", "Relief Society", "Charitable Trust", "Trading Company",
            "Logistics Ltd", "Exchange House", "Cultural Association", "Media Center",
            "Brotherhood Association", "Import-Export Group", "Welfare Committee"]
AGENCIES = ["national police", "counter-terrorism police", "state security service",
            "federal prosecutor's office", "gendarmerie", "intelligence directorate",
            "border guard", "financial intelligence unit", "military prosecutor"]
REAL_CITIES = {"en": ["Manchester", "Nairobi", "Lagos", "Toronto", "Mombasa", "Birmingham"],
               "ar": ["Mosul", "Tripoli", "Homs", "Benghazi", "Taiz", "Irbid"],
               "ru": ["Kazan", "Makhachkala", "Grozny", "Rostov-on-Don", "Nalchik", "Samara"],
               "fr": ["Marseille", "Lyon", "Toulouse", "Liège", "Bamako", "Niamey"],
               "es": ["Ceuta", "Melilla", "Valencia", "Bilbao", "Cúcuta", "Medellín"],
               "he": ["Ashdod", "Beersheba", "Netanya", "Sderot", "Eilat", "Acre"],
               "fa": ["Zahedan", "Ahvaz", "Kermanshah", "Herat", "Mashhad", "Tabriz"],
               "tr": ["Gaziantep", "Diyarbakır", "Adana", "Mersin", "Şanlıurfa", "Hatay"],
               "de": ["Dortmund", "Leipzig", "Bremen", "Graz", "Duisburg", "Zürich"]}
SYL = ["ka", "ro", "vel", "tan", "mar", "sol", "dra", "ven", "lis", "gor", "bel", "nor", "ash",
       "kel", "dor", "mir", "zan", "tor", "vas", "len", "kar", "sel", "bar", "ost", "ril", "dun"]

ATTACK_TARGETS = ["a police station", "an army convoy", "a fuel depot", "a border post",
                  "a market", "a power substation", "a government office", "a military base",
                  "a bus carrying soldiers", "a bridge", "a telecom tower", "a checkpoint"]
ATTACK_MEANS = ["a car bomb", "drones", "rockets", "an improvised explosive device", "gunmen",
                "mortar fire", "an arson attack", "a suicide bomber", "a roadside bomb"]
OFFENCES = ["recruiting fighters online", "financing the group", "planning an attack",
            "smuggling weapons", "running the group's propaganda channel",
            "membership in the group", "transferring money to the group", "buying drones"]
NEAR_MISS = [
    "a football match report full of words like 'attack', 'strike' and 'shot'",
    "a corporate hostile-takeover story",
    "a video game review about commanding rebels",
    "a film review of a thriller about a terrorist cell",
    "a history lesson about a 19th-century uprising",
    "a cybersecurity vendor's marketing text about 'threat actors'",
    "a chess tournament report with 'aggressive attacks'",
    "a report on a wildlife 'attack' by wolves on livestock",
    "a political campaign 'launching an offensive' on taxes",
    "a cooking competition with a 'bombshell' twist",
    "a medical article on a 'heart attack' study",
    "a boxing fight report",
]
IRRELEVANT = ["a recipe", "a weather forecast", "a travel guide paragraph", "a smartphone review",
              "a real-estate listing", "a gardening tip", "a cookie consent / website navigation "
              "text", "a school announcement", "a museum opening", "a stock-market summary"]


@dataclass
class Spec:
    id: str
    lang: str
    scenario: str
    length: str
    style: str
    facts: list[str]
    names: dict[str, str] = field(default_factory=dict)
    focus: str = "terrorism, armed groups, attacks, threats, financing"

    def as_dict(self) -> dict:
        return asdict(self)


class Sampler:
    def __init__(self, seed: int):
        self.r = random.Random(seed)
        self._used: set[str] = set()

    def pick(self, weights: dict[str, float]) -> str:
        return self.r.choices(list(weights), list(weights.values()))[0]

    def _unique(self, make) -> str:
        for _ in range(50):
            v = make()
            if v not in self._used:
                self._used.add(v)
                return v
        return v

    def person(self, lang: str) -> str:
        c = self.r.choice(CULTURES[lang])
        return self._unique(lambda: f"{self.r.choice(FIRST[c])} {self.r.choice(LAST[c])}")

    def group(self) -> str:
        return self._unique(lambda: (f"{self.r.choice(GROUP_ADJ)} {self.r.choice(GROUP_NOUN)} "
                                     f"{self.r.choice(GROUP_WORD)}"))

    def org(self) -> str:
        return self._unique(lambda: (f"{self.r.choice(GROUP_ADJ)} {self.r.choice(GROUP_NOUN)} "
                                     f"{self.r.choice(ORG_KIND)}"))

    def town(self, lang: str) -> str:
        if self.r.random() < 0.35:
            return self.r.choice(REAL_CITIES[lang])
        n = self.r.choice([2, 2, 3])
        return "".join(self.r.choice(SYL) for _ in range(n)).capitalize()

    def handle(self, base: str) -> str:
        stem = "".join(w[:4].lower() for w in base.split()[:2])
        return f"@{stem}{self.r.choice(['', '_media', '_news', '_official', str(self.r.randint(1, 99))])}"

    def domain(self, base: str) -> str:
        stem = "-".join(w.lower() for w in base.split()[:2])
        return f"{stem}{self.r.choice(['', '-info', '-news'])}.{self.r.choice(['org', 'net', 'info', 'com'])}"

    def wallet(self) -> str:
        kind = self.r.random()
        if kind < 0.6:   # bech32 alphabet, so it looks like a real address
            return "bc1q" + "".join(self.r.choice("023456789acdefghjklmnpqrstuvwxyz")
                                    for _ in range(38))
        if kind < 0.8:
            return "0x" + "".join(self.r.choice("0123456789abcdef") for _ in range(40))
        return "T" + "".join(self.r.choice(string.ascii_letters + string.digits)
                             for _ in range(33))

    def phone(self) -> str:
        return (f"+{self.r.choice([90, 961, 963, 964, 7, 33, 34, 49, 98, 972, 44, 254])} "
                f"{self.r.randint(100, 999)} {self.r.randint(100, 999)} {self.r.randint(1000, 9999)}")

    def date(self) -> str:
        y = self.r.choice([2024, 2025, 2025, 2026, 2026, 2026, 2027])
        return f"{self.r.randint(1, 28)} {self.r.choice(MONTHS)} {y}"

    def amount(self) -> str:
        return self.r.choice([f"{self.r.randint(2, 90)} bitcoin",
                              f"{self.r.randint(5, 900) * 1000:,} US dollars",
                              f"{self.r.randint(10, 500) * 1000:,} euros",
                              f"{self.r.randint(1, 50) * 1000:,} USDT"])

    # ------------------------------------------------------------------ scenarios

    def spec(self, i: int, lang_weights: dict[str, float] = LANG_WEIGHTS) -> Spec:
        lang = self.pick(lang_weights)
        scen = self.pick(SCENARIOS)
        s = Spec(id=f"syn-{i:05d}", lang=lang, scenario=scen, length=self.pick(LENGTHS),
                 style=self.r.choice(STYLES), facts=[])
        if self.r.random() < 0.25:
            s.focus = self.r.choice(["violent extremism and terrorist financing",
                                     "armed groups in the region", "online extremist propaganda",
                                     "terrorism threat assessment", "illicit finance networks"])
        getattr(self, f"_{scen}")(s)
        return s

    def _base(self, s: Spec, *, person=0, group=0, org=0, town=0) -> dict:
        n = {}
        for k in range(person):
            n[f"person{k + 1}"] = self.person(s.lang)
        for k in range(group):
            n[f"group{k + 1}"] = self.group()
        for k in range(org):
            n[f"org{k + 1}"] = self.org()
        for k in range(town):
            n[f"place{k + 1}"] = self.town(s.lang)
        s.names = n
        return n

    def _claim_attack(self, s: Spec) -> None:
        n = self._base(s, person=1, group=1, town=1)
        s.facts = [f"On {self.date()}, {self.r.choice(ATTACK_MEANS)} hit "
                   f"{self.r.choice(ATTACK_TARGETS)} in {n['place1']}; "
                   f"{self.r.randint(0, 14)} people were killed.",
                   f"The {n['group1']} claimed responsibility."]
        if self.r.random() < 0.6:
            s.facts.append(f"{n['person1']} is the {n['group1']}'s commander/leader.")
        else:
            s.facts.append(f"{n['person1']}, a police spokesperson, confirmed the toll.")
        if self.r.random() < 0.5:
            n["channel"] = self.handle(n["group1"])
            s.facts.append(f"The claim was posted on the group's Telegram channel {n['channel']}.")
        if self.r.random() < 0.3:
            s.facts.append("The group threatened further attacks.")

    def _attack_reported(self, s: Spec) -> None:
        n = self._base(s, person=1, group=1, town=1)
        s.facts = [f"An attack with {self.r.choice(ATTACK_MEANS)} on "
                   f"{self.r.choice(ATTACK_TARGETS)} in {n['place1']} on {self.date()}.",
                   f"According to {n['person1']}, a {self.r.choice(AGENCIES)} official, the "
                   f"{n['group1']} carried it out. No group has claimed it."]

    def _arrest(self, s: Spec) -> None:
        k = self.r.choice([1, 1, 2])
        n = self._base(s, person=k, group=1, town=1)
        who = " and ".join(n[f"person{j + 1}"] for j in range(k))
        s.facts = [f"The {self.r.choice(AGENCIES)} arrested {who} in {n['place1']} on "
                   f"{self.date()}."]
        if self.r.random() < 0.5:
            s.facts.append(f"They are confirmed members of the {n['group1']}, accused of "
                           f"{self.r.choice(OFFENCES)}.")
        else:
            s.facts.append(f"They are only SUSPECTED of links to the {n['group1']} "
                           f"(accused of {self.r.choice(OFFENCES)}); membership is not proven.")

    def _finance(self, s: Spec) -> None:
        n = self._base(s, person=1, group=1, org=1, town=1)
        n["wallet"] = self.wallet()
        s.facts = [f"The {n['org1']} sent {self.amount()} to the {n['group1']} via the "
                   f"crypto wallet {n['wallet']}.",
                   f"{n['person1']} is the {n['group1']}'s finance chief and controls the wallet."]
        if self.r.random() < 0.4:
            s.facts.append(f"{n['person1']} was arrested in {n['place1']} on {self.date()}.")
        if self.r.random() < 0.3:
            n["wallet2"] = self.wallet()
            s.facts.append(f"The funds were then moved to a second wallet, {n['wallet2']}.")

    def _infrastructure(self, s: Spec) -> None:
        n = self._base(s, person=1, group=1)
        kinds = self.r.sample(["channel", "domain", "email", "phone", "account"],
                              self.r.choice([1, 2, 3]))
        s.facts = []
        for kind in kinds:
            if kind == "channel":
                n["channel"] = self.handle(n["group1"])
                s.facts.append(f"The {n['group1']} runs the Telegram channel {n['channel']}.")
            elif kind == "domain":
                n["domain"] = self.domain(n["group1"])
                s.facts.append(f"The group operates the website {n['domain']}.")
            elif kind == "email":
                n["email"] = f"{self.r.choice(['contact', 'press', 'info', 'media'])}@" + \
                    self.domain(n["group1"])
                s.facts.append(f"It uses the email address {n['email']} for press contact.")
            elif kind == "phone":
                n["phone"] = self.phone()
                s.facts.append(f"{n['person1']} gave the phone number {n['phone']} "
                               "for donations/contact.")
            else:
                n["account"] = self.handle(n["person1"])
                s.facts.append(f"{n['person1']} posts on X under the account {n['account']}.")
        s.facts.append(self.r.choice([
            f"{n['person1']} is the group's media official.",
            f"On {self.date()} the channel/site published a new propaganda video.",
            "The platform suspended the account this week.",
        ]))

    def _meeting(self, s: Spec) -> None:
        n = self._base(s, person=2, group=1, town=1)
        s.facts = [f"{n['person1']} met {n['person2']} in {n['place1']} on {self.date()}.",
                   f"{n['person1']} is a known leader of the {n['group1']}. [Say nothing about "
                   f"whether {n['person2']} belongs to any group.]"]

    def _leadership(self, s: Spec) -> None:
        n = self._base(s, person=2, group=1, town=1)
        s.facts = [f"{n['person1']} became the new leader of the {n['group1']} after "
                   f"{n['person2']} was killed in {n['place1']}.",
                   f"The group is based in {n['place1']}."]

    def _affiliation(self, s: Spec) -> None:
        n = self._base(s, person=1, group=1, org=1)
        s.facts = [f"The {n['org1']} is openly affiliated with the {n['group1']} and acts as "
                   "its civilian front.",
                   f"{n['person1']}, the {n['org1']}'s director, regularly communicates with "
                   f"the group's leadership."]

    def _threat(self, s: Spec) -> None:
        n = self._base(s, person=1, group=1, town=1)
        n["channel"] = self.handle(n["group1"])
        s.facts = [f"The {n['group1']} threatened to attack {self.r.choice(ATTACK_TARGETS)} "
                   f"in {n['place1']} in a message posted on {n['channel']} on {self.date()}.",
                   f"The message was signed by its spokesman {n['person1']}."]

    def _denial(self, s: Spec) -> None:
        n = self._base(s, person=1, group=1, org=1, town=1)
        s.facts = [self.r.choice([
            f"{n['person1']} DENIED being a member of the {n['group1']}.",
            f"The {n['org1']} DENIED sending any money to the {n['group1']}.",
            f"Officials in {n['place1']} REJECTED reports that the {n['group1']} was behind "
            f"a recent bombing, calling them speculation.",
            f"The {n['group1']} DENIED any link to {n['person1']}, who was arrested in "
            f"{n['place1']}.",
        ])]

    def _speculative(self, s: Spec) -> None:
        n = self._base(s, person=1, group=2, org=1)
        s.facts = [self.r.choice([
            f"Analysts SUSPECT that the {n['group1']} MAY be receiving money from the "
            f"{n['org1']}, but no transfer has been confirmed.",
            f"Rumours circulate that {n['person1']} could be the new leader of the "
            f"{n['group1']}; this is unconfirmed.",
            f"It is unclear whether the {n['group1']} and the {n['group2']} have merged; "
            "some observers speculate so.",
            f"Some media ask whether {n['person1']} is secretly a member of the "
            f"{n['group1']}; there is no evidence.",
        ])]

    def _mixed_stance(self, s: Spec) -> None:
        n = self._base(s, person=2, group=1, org=1, town=1)
        s.facts = [f"{n['person1']} was confirmed as the {n['group1']}'s leader.",
                   f"{n['person2']} DENIED any role in the group and said he only knew "
                   f"{n['person1']} from school.",
                   f"Prosecutors SUSPECT, without proof, that the {n['org1']} funds the group."]

    def _near_miss(self, s: Spec) -> None:
        n = self._base(s, person=1, town=1)
        s.facts = [f"Topic: {self.r.choice(NEAR_MISS)}. [It must not be about terrorism or "
                   "armed groups, though the vocabulary may sound violent or militant.]",
                   f"Mention {n['person1']} and {n['place1']}."]

    def _irrelevant(self, s: Spec) -> None:
        n = self._base(s, town=1)
        s.facts = [f"Topic: {self.r.choice(IRRELEVANT)}, mentioning {n['place1']}. [Nothing "
                   "about security, violence or politics.]"]


MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


def sample(n: int, seed: int = 7, start: int = 0, langs: list[str] | None = None) -> list[Spec]:
    """`langs` restricts the language (for top-ups of languages the filters thinned out)."""
    s = Sampler(seed)
    weights = {k: v for k, v in LANG_WEIGHTS.items() if not langs or k in langs}
    out = []
    for i in range(start, start + n):
        spec = s.spec(i, weights)
        out.append(spec)
    return out


def writer_prompt(s: Spec) -> str:
    lang = LANGS[s.lang]
    names = "\n".join(f"- {k}: {v}" for k, v in s.names.items())
    script = ""
    if s.lang in ("ar", "ru", "he", "fa"):
        script = (" Write person names and place names in the language's own script "
                  "(transliterate them).")
    return f"""Write {s.style} in {lang}. Length: {s.length}. It is FICTIONAL training material: \
invented people, groups and events.

Facts to convey (in your own words; keep stance exactly as stated: confirmed facts as facts, \
denials as denials, suspicions as suspicions):
{chr(10).join('- ' + f for f in s.facts)}

Names to use:
{names}

Rules:
- Write natural {lang}. Translate group and organization names into {lang} if that is \
natural.{script}
- Text in [square brackets] is guidance for you: follow it, never write it.
- Copy handles, website domains, email addresses, phone numbers and wallet addresses \
character for character.
- You may add plausible minor detail, but no other named groups.
- Output only the text itself: no title line, no notes, no quotation marks around it."""

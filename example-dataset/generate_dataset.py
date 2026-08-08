"""
Antisemitism Dataset Generator
Generates 1000 labelled Telegram-style posts using seed + variation.
Outputs: antisemitism_dataset.json and antisemitism_dataset.csv
"""

import csv
import hashlib
import json
import random
from datetime import datetime, timedelta

random.seed(42)

# ── Label definitions ────────────────────────────────────────────────────────

IHRA_LABELS = [
    "Calling for, aiding, or justifying the killing or harming of Jews",            # 0
    "Mendacious, dehumanizing, demonizing, or stereotypical allegations about Jews", # 1
    "Accusing Jews as a people of being responsible for real or imagined wrongdoing",# 2
    "Denying the fact, scope, mechanisms, or intentionality of the Holocaust",       # 3
    "Accusing the Jews as a people, or Israel as a state, of inventing or exaggerating the Holocaust", # 4
    "Accusing Jewish citizens of dual loyalty",                                      # 5
    "Applying double standards to Israel not expected of any other democratic nation",# 6
    "Using symbols and images associated with classic antisemitism to characterize Israel or Israelis", # 7
    "Drawing comparisons of contemporary Israeli policy to that of the Nazis",       # 8
    "Holding Jews collectively responsible for actions of the State of Israel",      # 9
]

KEYWORD_LABELS = [
    "Holocaust",         # 0
    "October 7th",       # 1
    "Zionist",           # 2
    "synagogue",         # 3
    "antisemitism",      # 4
    "Jewish conspiracy", # 5
    "globalists",        # 6
    "media control",     # 7
    "bank control",      # 8
    "dual loyalty",      # 9
]

TIER_NAMES = {
    0: "clean",
    1: "subtle",
    2: "coded",
    3: "explicit",
    4: "hostile",
}

SCORE_RANGES = {
    0: (0.03, 0.20),
    1: (0.22, 0.44),
    2: (0.48, 0.67),
    3: (0.68, 0.87),
    4: (0.88, 0.99),
}

# ── Variation helpers ────────────────────────────────────────────────────────

COUNTRIES = ["Germany", "France", "the United States", "the United Kingdom",
             "Poland", "Hungary", "Russia", "Ukraine", "Argentina", "Canada", "Sweden"]
CITIES    = ["Berlin", "Paris", "New York", "London", "Warsaw", "Budapest",
             "Moscow", "Kyiv", "Buenos Aires", "Toronto", "Stockholm"]
NAMES     = ["Aaron", "David", "Sarah", "Rachel", "Michael", "Hannah",
             "Jacob", "Miriam", "Benjamin", "Leah"]
AMOUNTS   = ["billions", "trillions", "hundreds of billions"]
ORGS      = ["AIPAC", "the ADL", "the World Jewish Congress", "B'nai B'rith"]

def _r(lst): return random.choice(lst)

# Inline variation: swap one phrase to add linguistic diversity without
# changing the meaning or the ground-truth label.
SWAP_PAIRS = [
    ("Jewish people", "Jews"),
    ("Israelis", "the Israeli government"),
    ("the media", "mainstream media"),
    ("globalists", "global elites"),
    ("bankers", "financiers"),
    ("Zionists", "Zionist forces"),
    ("synagogue", "Jewish community centre"),
    ("Holocaust", "Shoah"),
    ("antisemitism", "Jew-hatred"),
    ("Israel", "the Zionist state"),
]

def _vary_text(text: str, n: int) -> list[str]:
    """Produce up to n textual variants of a seed, including the seed itself."""
    results = [text]
    for a, b in random.sample(SWAP_PAIRS, min(n - 1, len(SWAP_PAIRS))):
        if a in text:
            results.append(text.replace(a, b, 1))
        elif b in text:
            results.append(text.replace(b, a, 1))
        if len(results) >= n:
            break
    # pad with slight punctuation / emoji suffix variants
    suffixes = ["", " 👀", " 🔥", " ‼️", " 📢", " 🤔", "!!", " — share this!", " — wake up!"]
    while len(results) < n:
        results.append(text + random.choice(suffixes))
    return results[:n]

# ── Seed definitions ─────────────────────────────────────────────────────────
# Schema: (text, tier, ihra_indices, keyword_indices, country_or_None, sentiment)
# Counts: tier0=42, tier1=42, tier2=42, tier3=42, tier4=32  → 200 seeds

SEEDS = [

    # ════════════════════════════════════════════════════════════════════
    # TIER 0 — CLEAN  (42 seeds × ~6 = 252 posts)
    # ════════════════════════════════════════════════════════════════════

    # Jewish culture & holidays
    ("Shabbat Shalom everyone! Wishing a peaceful and restful Shabbat to all those celebrating tonight. 🕎", 0, [], [], None, "Supportive"),
    ("Chag Sameach! Wishing everyone a joyful and meaningful Passover. May the story of freedom inspire us all.", 0, [], [], None, "Supportive"),
    ("Hanukkah starts tonight! Eight nights of light, family, and latkes. 🕯️🕯️", 0, [], [], None, "Supportive"),
    ("Rosh Hashanah tonight. L'Shana Tova to all marking the Jewish New Year 🍎🍯", 0, [], [], None, "Supportive"),
    ("Yom Kippur begins at sundown. An important day of reflection and atonement for Jewish communities worldwide.", 0, [], [], None, "Neutral"),
    ("Just visited a beautiful synagogue in Prague. The history and architecture are breathtaking.", 0, [], [3], "Czech Republic", "Supportive"),
    ("The Jewish Museum in Berlin is one of the most powerful museums I've ever visited. A must-see.", 0, [], [], "Germany", "Supportive"),
    ("Klezmer music is so joyful and moving. Attended a live performance last night — incredible.", 0, [], [], None, "Supportive"),
    ("Making matzah ball soup from scratch this year for Passover. Wish me luck 🍜", 0, [], [], None, "Supportive"),
    ("Celebrated Purim with my neighbors today. The costumes were amazing and the hamantaschen were delicious!", 0, [], [], None, "Supportive"),

    # Israel positive/neutral news
    ("Israel's drip-irrigation technology is helping farmers across Africa and Asia grow food with minimal water. Remarkable innovation.", 0, [], [], "Israel", "Supportive"),
    ("Tel Aviv ranked among the top tech startup ecosystems in the world again this year. 🇮🇱", 0, [], [], "Israel", "Supportive"),
    ("Israeli scientists have published a major breakthrough in Alzheimer's research. Great news for patients worldwide.", 0, [], [], "Israel", "Supportive"),
    ("Just landed in Tel Aviv for the first time. The food, the energy, the people — absolutely love it here.", 0, [], [], "Israel", "Supportive"),
    ("The Dead Sea is truly unlike anything else on earth. Floating effortlessly while reading a newspaper 📰", 0, [], [], "Israel", "Supportive"),
    ("Israel's Magen David Adom volunteers continue providing aid across disaster zones worldwide. Proud of this work.", 0, [], [], "Israel", "Supportive"),
    ("Israel and the UAE normalizing relations is a historic step for regional stability. Hope it continues.", 0, [], [], "Israel", "Neutral"),

    # Holocaust remembrance & education
    ("Today is International Holocaust Remembrance Day. Never forget the six million Jewish lives lost. #NeverAgain", 0, [], [0, 4], None, "Neutral"),
    ("Visited Yad Vashem in Jerusalem today. Walked through in silence. Some things cannot be put into words.", 0, [], [0], "Israel", "Neutral"),
    ("My grandmother survived Auschwitz. She passed last year at 96. Her testimony will live on forever. #HolocaustRemembrance", 0, [], [0], "Poland", "Neutral"),
    ("Schools should make Holocaust education mandatory. Understanding history is the only way to prevent it repeating.", 0, [], [0, 4], None, "Neutral"),
    ("The last surviving Nuremberg prosecutor has died at 103. A life devoted to justice. Rest in peace.", 0, [], [0], "United States", "Neutral"),
    ("Reading 'Night' by Elie Wiesel for the third time. Devastating and essential. Everyone should read this book.", 0, [], [0], None, "Neutral"),
    ("The Auschwitz-Birkenau Memorial receives over two million visitors a year. Memory is an act of resistance.", 0, [], [0], "Poland", "Neutral"),

    # Counter-antisemitism & solidarity
    ("Antisemitic incidents have risen sharply this year. This is unacceptable and we all must speak out.", 0, [], [4], None, "Negative"),
    ("A synagogue in my city was vandalized last night. The whole community stands with our Jewish neighbors.", 0, [], [3, 4], None, "Neutral"),
    ("Holocaust denial is not an opinion. It is a lie and an act of hate. Full stop.", 0, [], [0, 4], None, "Negative"),
    ("Proud to stand with the Jewish community at today's solidarity rally. Hate has no place here.", 0, [], [4], None, "Supportive"),
    ("Reporting a wave of antisemitic messages on this platform. Please report and block these accounts.", 0, [], [4], None, "Neutral"),
    ("Jewish students deserve to feel safe on campus. Universities must do more to combat antisemitism.", 0, [], [4], None, "Neutral"),

    # Completely unrelated topics (ensures clean tier is genuinely clean)
    ("The new season of this show is absolutely incredible. Cannot stop watching.", 0, [], [], None, "Supportive"),
    ("Traffic in the city this morning was a nightmare. Took two hours for what's normally 30 minutes.", 0, [], [], None, "Negative"),
    ("Beautiful sunrise this morning. Took a long walk by the river. Sometimes you just need that.", 0, [], [], None, "Supportive"),
    ("The economic data released today shows inflation easing slightly. Still a long way to go though.", 0, [], [], None, "Neutral"),
    ("Climate change summit opens this week. Leaders need to stop talking and start acting.", 0, [], [], None, "Negative"),
    ("My team won the championship last night!! What a match. Still can't believe it 🏆", 0, [], [], None, "Supportive"),
    ("Local election results are in. Turnout was low as expected. We need to do better.", 0, [], [], None, "Neutral"),
    ("The new AI tools coming out this year are genuinely impressive. Exciting and slightly terrifying.", 0, [], [], None, "Neutral"),
    ("Best coffee I've ever had was at a tiny café in Porto. If you're ever there, go.", 0, [], [], "Portugal", "Supportive"),
    ("Healthcare workers deserve better pay and conditions. Simple as that.", 0, [], [], None, "Neutral"),
    ("Just finished a 5km run for the first time. Legs are dead but feeling great 🏃", 0, [], [], None, "Supportive"),
    ("Film recommendation: incredible documentary about ocean conservation. Everyone should watch.", 0, [], [], None, "Supportive"),

    # ════════════════════════════════════════════════════════════════════
    # TIER 1 — SUBTLE  (42 seeds × ~5 = 210 posts, trimmed to 200)
    # ════════════════════════════════════════════════════════════════════

    # Dog whistles & coded language
    ("Why do the globalists always push for open borders? Who benefits from this chaos? Just asking.", 1, [], [6], None, "Hostile"),
    ("The global elites pulling the strings behind our governments — follow the money. Always follow the money.", 1, [], [6, 8], None, "Hostile"),
    ("Funny how the same faces keep appearing at Davos every year. The same bankers, the same agenda.", 1, [], [6, 8], None, "Hostile"),
    ("Hollywood and the major news networks all seem to push the same narrative. Ever wonder why?", 1, [], [7], None, "Hostile"),
    ("The Federal Reserve has never been properly audited. Who really controls our money supply?", 1, [], [8], "United States", "Hostile"),
    ("(((They))) don't want you to know the truth. That's why the algorithm keeps suppressing this content.", 1, [], [5], None, "Hostile"),
    ("Notice how every time a politician criticizes Israel, they suddenly face a scandal or career problems?", 1, [], [9], None, "Hostile"),
    ("The think tanks shaping foreign policy — look at who funds them. You'll see a pattern.", 1, [], [5, 8], None, "Hostile"),
    ("Not saying who, but a certain tiny group of people owns a disproportionate amount of our media.", 1, [], [7], None, "Hostile"),
    ("George Soros funds organizations in over 100 countries. Why does one man have so much global influence?", 1, [], [6], "United States", "Hostile"),

    # "Just asking questions"
    ("I'm not antisemitic but why do Jewish people make up such a small percentage of the population yet dominate so many industries?", 1, [1], [5], None, "Negative"),
    ("Legitimate question: why is it taboo to even ask who benefits from US foreign policy in the Middle East?", 1, [], [9], "United States", "Negative"),
    ("Just curious — why are most of the major investment banks run by people with the same background?", 1, [], [8], None, "Negative"),
    ("Why does the mainstream media never critically cover Israel the way it covers other countries? Genuinely asking.", 1, [], [7], None, "Negative"),
    ("I'm allowed to ask: why do so many government advisors hold dual citizenship with Israel?", 1, [], [9], None, "Negative"),
    ("No hate intended, but has anyone else noticed which communities seem to be behind most major financial scandals?", 1, [1], [8], None, "Negative"),
    ("Why is it antisemitic to point out that AIPAC has enormous influence over US elections? That's just factual.", 1, [], [9], "United States", "Negative"),
    ("Question: why does every Western country prioritize Israel's security over the welfare of its own citizens?", 1, [], [9], None, "Negative"),

    # Suspicious "coincidences"
    ("Interesting coincidence that so many of the architects of the 2008 financial crash came from the same institutions…", 1, [], [8], "United States", "Neutral"),
    ("Strange coincidence how the Ukraine war benefited the same oligarchs who were already connected to Washington.", 1, [], [6], "Ukraine", "Neutral"),
    ("What a coincidence that the biggest media donors all seem to have the same political interests in the Middle East.", 1, [], [7, 9], None, "Neutral"),
    ("Remarkable how the same foundation keeps appearing connected to every major social upheaval of the last 20 years.", 1, [], [6], None, "Neutral"),
    ("Another 'coincidence' — the politicians who voted for the bailout all received campaign donations from the same banks.", 1, [], [8], None, "Neutral"),
    ("It's just a coincidence that the loudest voices for censorship are the ones who'd benefit most from it.", 1, [], [5], None, "Neutral"),

    # Soft dual loyalty
    ("American politicians putting Israel's interests ahead of American ones. When does that become a problem?", 1, [], [9], "United States", "Negative"),
    ("A congressman with dual citizenship voting on military aid to the country he holds citizenship in. That's fine?", 1, [5], [9], None, "Negative"),
    ("Loyalty to a foreign state should disqualify you from public office. That applies to everyone — no exceptions.", 1, [5], [9], None, "Negative"),
    ("Why do we allow foreign lobbies to dictate our domestic policy? AIPAC spends more than any other foreign lobby.", 1, [5], [9], "United States", "Negative"),
    ("Some people serve two masters. The question is which one comes first when there's a conflict.", 1, [5], [9], None, "Negative"),

    # "I'm not antisemitic but…"
    ("I have nothing against Jewish people, but you have to admit there's a reason they've been expelled from countries throughout history.", 1, [1, 2], [], None, "Hostile"),
    ("Not a hateful person, but when the same group appears at the top of media, finance, and politics, you have to ask why.", 1, [1], [7, 8], None, "Hostile"),
    ("I love Jewish culture and people, but Israel's actions make it hard to distinguish between Israeli policy and broader Jewish influence.", 1, [9], [], "Israel", "Neutral"),
    ("Criticism of Israel isn't antisemitism. And sometimes what looks like antisemitism is just people connecting the dots.", 1, [2], [], None, "Neutral"),
    ("I don't hate anyone, but I do think certain communities prioritize their own interests over national ones — and that needs discussion.", 1, [5], [], None, "Neutral"),
    ("People are afraid to say it openly, but they think it privately. That gap between public and private opinion is growing.", 1, [], [5], None, "Neutral"),
    ("Not blaming anyone, just observing patterns. Patterns don't lie even when people say you're not allowed to discuss them.", 1, [], [5], None, "Neutral"),

    # ════════════════════════════════════════════════════════════════════
    # TIER 2 — CODED / CROSSING LINE  (42 seeds × ~5 = 210, trimmed to 200)
    # ════════════════════════════════════════════════════════════════════

    # Anti-Zionism crossing into antisemitism
    ("Zionism is a racist ideology. Jewish people who support it are complicit in ethnic cleansing.", 2, [9], [2, 4], "Israel", "Hostile"),
    ("Every Jew who stays silent about Gaza is personally complicit in genocide. Silence is collaboration.", 2, [9], [2], "Israel", "Hostile"),
    ("The Zionist lobby has America by the throat. It's not a conspiracy — it's just money and power.", 2, [2], [2, 8], "United States", "Hostile"),
    ("Jewish Zionists have infected Western institutions. Their goal is to silence all criticism of Israel.", 2, [2], [2], None, "Hostile"),
    ("Zionism is to Judaism what Nazism is to Germany — an extremist political movement hiding behind identity.", 2, [8], [2], None, "Hostile"),
    ("Israel could not exist for a single day without Jewish financial networks propping it up globally.", 2, [2], [2, 8], "Israel", "Hostile"),
    ("The apartheid state of Israel is maintained by Jewish donors worldwide. They all share responsibility.", 2, [9], [2], "Israel", "Hostile"),
    ("Anti-Zionism is not antisemitism. But holding every Jew responsible for Israel IS a form of anti-Zionism. And it's justified.", 2, [9], [2, 4], None, "Hostile"),

    # Conspiracy adjacent
    ("COVID vaccines were designed by people connected to the Davos crowd — same names, same agenda, same profits.", 2, [2], [6], None, "Hostile"),
    ("Klaus Schwab, Bill Gates, George Soros — the 'Great Reset' crowd. Look at who funds them. Same people every time.", 2, [2], [5, 6], None, "Hostile"),
    ("The media, the banks, and the pharmaceuticals are all controlled by the same networks. Connect the dots.", 2, [2], [7, 8], None, "Hostile"),
    ("9/11 benefited a very specific set of people. Those people had advance knowledge. This is documented, not conspiracy.", 2, [2], [5], "United States", "Hostile"),
    ("Jewish organizations are the biggest funders of mass immigration into Europe. This is not a theory — look at the tax filings.", 2, [2], [5, 6], "Germany", "Hostile"),
    ("George Soros has spent over $30 billion dismantling national sovereignty worldwide. Who gave him that mission?", 2, [2], [6], "United States", "Hostile"),
    ("Every major color revolution in the last 20 years traces back to the same NGO networks. Follow the funding.", 2, [2], [5, 6], None, "Hostile"),

    # Holocaust minimization (not full denial)
    ("Six million is a very round number. Historians debate the actual toll — why is that discussion forbidden?", 2, [3, 4], [0], None, "Hostile"),
    ("More civilians died in the Allied bombing of Dresden than in some camps. Why is one tragedy commemorated and the other ignored?", 2, [3], [0], "Germany", "Negative"),
    ("The Holocaust was a tragedy, but so were the famines in India and the gulags. Why does one event dominate all others?", 2, [3], [0], None, "Neutral"),
    ("Hollywood has made the Holocaust its permanent focus. Almost as if some people profit from keeping the guilt alive.", 2, [4], [0], "United States", "Hostile"),
    ("Why is questioning Holocaust figures illegal in so many countries? Real historical events don't need legal protection.", 2, [3, 4], [0], None, "Hostile"),

    # Dual loyalty accusations
    ("Jewish politicians in Germany consistently vote for Israel's interests over Germany's. This is dual loyalty — call it what it is.", 2, [5], [9], "Germany", "Hostile"),
    ("Jewish journalists write about antisemitism constantly but never about Jewish power in media. Conflict of interest?", 2, [5], [7], None, "Negative"),
    ("American Jews donate to Israeli defense funds while living safely in the US. Their true loyalty is clear.", 2, [5], [9], "United States", "Hostile"),
    ("Dual citizenship in Israel should disqualify anyone from serving in Western governments. Period.", 2, [5], [9], None, "Negative"),

    # Nazi comparisons / collective responsibility
    ("Israel's siege of Gaza is a textbook example of collective punishment — the same tactic used in the Warsaw Ghetto.", 2, [8, 9], [0, 2], "Israel", "Negative"),
    ("What Israel is doing to Palestinians is exactly what was done to Jews in the 1930s. They became what they fought.", 2, [8], [0], "Israel", "Negative"),
    ("The IDF's behavior in civilian areas is indistinguishable from Nazi military doctrine. History repeating itself.", 2, [8], [0, 2], "Israel", "Hostile"),
    ("Jewish communities in the West cheer on Israeli airstrikes. Collective enthusiasm, collective responsibility.", 2, [9], [], None, "Hostile"),
    ("Every Jewish donor to Israeli military campaigns shares moral responsibility for the deaths of children.", 2, [9], [], "Israel", "Hostile"),
    ("Western Jews lobbying for more weapons to Israel while condemning civilian deaths — hypocrisy on a historic scale.", 2, [9], [2], None, "Negative"),
    ("The same people who lecture the world on the Holocaust are now running an open-air prison. Ironic, isn't it?", 2, [8, 9], [0], "Israel", "Negative"),

    # ════════════════════════════════════════════════════════════════════
    # TIER 3 — EXPLICIT ANTISEMITISM  (42 seeds × ~5 = 210, trimmed to 200)
    # ════════════════════════════════════════════════════════════════════

    # Stereotypes stated as fact
    ("Jews control the banks, the media, and the governments of the Western world. This is not a conspiracy — it's documented.", 3, [1, 2], [5, 7, 8], None, "Hostile"),
    ("Jewish people have a genetic disposition toward deception and manipulation. History proves this across civilizations.", 3, [1], [5], None, "Hostile"),
    ("The reason Jews are so successful is because they nepotistically promote each other while excluding everyone else.", 3, [1], [5], None, "Hostile"),
    ("Jews invented communism to destroy Christian civilization. Marx, Lenin, Trotsky — all Jewish. Look it up.", 3, [1, 2], [5], None, "Hostile"),
    ("Every major war in the last 200 years was engineered by Jewish bankers to profit from both sides.", 3, [1, 2], [5, 8], None, "Hostile"),
    ("Jewish lawyers dominate the legal system specifically to protect Jewish criminals from prosecution.", 3, [1, 2], [5], None, "Hostile"),
    ("The reason housing is unaffordable is Jewish landlords and real-estate firms systematically pricing out gentiles.", 3, [1, 2], [5, 8], None, "Hostile"),
    ("Jewish academics have infiltrated universities to promote degeneracy and erase Western culture.", 3, [1, 2], [5], None, "Hostile"),
    ("They own the entertainment industry and use it to normalize moral decay — this is a deliberate agenda.", 3, [2], [5, 7], None, "Hostile"),
    ("Every pharmaceutical company that profited from COVID was founded or run by Jews. Coincidence?", 3, [1, 2], [5], None, "Hostile"),

    # Holocaust denial
    ("The Holocaust is the biggest lie of the 20th century. Six million Jews did not die — the Red Cross documented far fewer.", 3, [3, 4], [0], None, "Hostile"),
    ("There were no gas chambers at Auschwitz. This has been proven by multiple engineers and chemists.", 3, [3], [0], "Poland", "Hostile"),
    ("The Holocaust was Allied war propaganda to justify the destruction of Germany. The real victims were Germans.", 3, [3, 4], [0], "Germany", "Hostile"),
    ("Jewish historians exaggerated Holocaust numbers to justify the creation of Israel. It's a founding myth.", 3, [3, 4], [0], "Israel", "Hostile"),
    ("Ask yourself: why is Holocaust denial illegal? Because the story can't survive scrutiny. That's why.", 3, [3, 4], [0], None, "Hostile"),
    ("Anne Frank's diary was written in ballpoint pen — which wasn't invented until after the war. Think about it.", 3, [4], [0], None, "Hostile"),
    ("The six million figure appeared in Jewish newspapers as early as 1900. It's a ritual number, not a historical one.", 3, [3, 4], [0], None, "Hostile"),
    ("Zyklon B was a delousing agent. The idea it was used for mass murder is physically and logistically impossible.", 3, [3], [0], None, "Hostile"),

    # Conspiracy theories naming Jews explicitly
    ("The Rothschild family controls every central bank on earth and has engineered every major financial crisis since 1815.", 3, [1, 2], [5, 8], None, "Hostile"),
    ("The Protocols of the Elders of Zion is not a forgery — it's a blueprint that has been executed with precision.", 3, [1, 2], [5], None, "Hostile"),
    ("Jewish billionaires funded Black Lives Matter to pit Americans against each other and prevent white unity.", 3, [2], [5, 6], "United States", "Hostile"),
    ("The Jewish lobby controls the US Congress. Every major military intervention since 1990 was done for Israel.", 3, [2], [5, 9], "United States", "Hostile"),
    ("Jews started World War I, World War II, and are now engineering World War III. Same families, same playbook.", 3, [2], [5], None, "Hostile"),
    ("The Frankfurt School was a Jewish intellectual project to destroy Western Christian civilization from within.", 3, [1, 2], [5], "Germany", "Hostile"),
    ("Jewish NGOs are deliberately funding mass migration into Europe to replace the native population.", 3, [2], [5, 6], "Germany", "Hostile"),
    ("The LGBT agenda is a Jewish invention to weaken traditional family structures in Christian societies.", 3, [2], [5], None, "Hostile"),

    # Dehumanization
    ("Jews are parasites who feed on host nations until there is nothing left, then move on. History confirms this.", 3, [1], [5], None, "Hostile"),
    ("The Talmud literally instructs Jews to cheat, lie to, and exploit non-Jews. This is their sacred text.", 3, [1], [5], None, "Hostile"),
    ("Judaism is not a religion, it's a supremacist ideology disguised as religion. Jewish people are trained from birth to believe they are chosen above all others.", 3, [1], [5], None, "Hostile"),
    ("Jews have no genuine home. They are eternal wanderers who corrupt every nation they settle in.", 3, [1], [5], None, "Hostile"),
    ("A Jew will smile at your face and stab you in the back. This is their nature — documented throughout history.", 3, [1], [5], None, "Hostile"),
    ("Jewish identity is built on victimhood and manipulation. The 'antisemitism' card is their most powerful weapon.", 3, [1], [5], None, "Hostile"),

    # IHRA-specific violations
    ("Displaying the Star of David alongside the Nazi swastika is the most accurate map of Israeli foreign policy.", 3, [7], [2], "Israel", "Hostile"),
    ("Israel is a settler-colonial project no different from apartheid South Africa — but worse, because it has nuclear weapons.", 3, [6, 8], [2], "Israel", "Hostile"),
    ("If you criticize Saudi Arabia you're brave. If you criticize Israel you lose your job. That double standard is Jewish power.", 3, [6], [7], None, "Hostile"),
    ("Every Jewish person in France must answer for Israel's actions. You cannot be Jewish and neutral.", 3, [9], [], "France", "Hostile"),
    ("Jewish community leaders in Germany have more loyalty to Tel Aviv than to Berlin. Always have, always will.", 3, [5], [9], "Germany", "Hostile"),
    ("October 7th was a response to 75 years of occupation. You cannot condemn resistance without condemning the occupation first.", 3, [], [1], "Israel", "Hostile"),

    # ════════════════════════════════════════════════════════════════════
    # TIER 4 — HOSTILE / VIOLENT  (32 seeds × ~5 = 160, trimmed to 150)
    # ════════════════════════════════════════════════════════════════════

    # Incitement to violence
    ("The only solution to the Jewish problem is the same one that has been tried many times throughout history.", 4, [0], [5], None, "Hostile"),
    ("Every synagogue in Europe should be shut down. They are command centres for a fifth column.", 4, [0], [3, 5], "Germany", "Hostile"),
    ("When the revolution comes, the bankers will be the first ones up against the wall. You know who they are.", 4, [0], [8], None, "Hostile"),
    ("Hitler was right about one thing. He identified the enemy clearly. We should have the courage to do the same.", 4, [0, 1], [5], "Germany", "Hostile"),
    ("Jews in positions of power must be removed. Peacefully if possible, but removed either way.", 4, [0], [5], None, "Hostile"),
    ("A final accounting is coming. The names are being written down. Every Rothschild, every Soros, every collaborator.", 4, [0], [5, 8], None, "Hostile"),
    ("Armed resistance against Zionist occupation is not terrorism — it is a sacred duty for every free person.", 4, [0], [2], "Israel", "Hostile"),
    ("Jewish journalists who push war propaganda are legitimate military targets. They are combatants.", 4, [0], [7], None, "Hostile"),

    # Genocidal rhetoric
    ("The world would be a better place without Israel and the people who created and defend it.", 4, [0], [2], "Israel", "Hostile"),
    ("Finishing what Hitler started is not hate speech — it is a political position held by millions.", 4, [0, 1], [5], None, "Hostile"),
    ("From the river to the sea — and after that, we deal with the Jewish diaspora that supported this colonial project.", 4, [0, 9], [2], "Israel", "Hostile"),
    ("Every Israeli civilian is a soldier. There are no innocent Zionists. None.", 4, [0, 9], [2, 1], "Israel", "Hostile"),
    ("Ethnic cleansing works both ways. Palestinians should practice what they preach — starting now.", 4, [0], [], "Israel", "Hostile"),
    ("The Nazis failed because they were too merciful. We should not make the same mistake.", 4, [0, 1], [5], None, "Hostile"),
    ("Death to Israel and all who fund it. This is not a threat — it is a prayer.", 4, [0], [2], "Israel", "Hostile"),
    ("Nuke Tel Aviv and the problem solves itself. The rest of the world can finally breathe.", 4, [0], [2], "Israel", "Hostile"),

    # Extreme fake news
    ("BREAKING: Declassified documents confirm Israel knew about the 9/11 attacks 72 hours in advance and said nothing.", 4, [2], [5], "United States", "Hostile"),
    ("CONFIRMED: Jewish doctors are deliberately administering different (weaker) vaccines to non-Jewish patients.", 4, [1, 2], [5], None, "Hostile"),
    ("LEAKED: Mossad operatives were behind the Nord Stream pipeline bombing. German intelligence has the proof.", 4, [2], [5], "Germany", "Hostile"),
    ("EXCLUSIVE: The IDF deliberately harvests organs from Palestinian civilians killed in airstrikes.", 4, [1], [2], "Israel", "Hostile"),
    ("BOMBSHELL: Rothschild family met privately with three heads of state one week before the 2008 financial crash.", 4, [2], [5, 8], "United States", "Hostile"),
    ("PROOF: The October 7th attack was a false flag staged by Mossad to justify the invasion of Gaza.", 4, [2, 4], [1, 5], "Israel", "Hostile"),

    # Glorifying October 7th & violence against Jews
    ("October 7th was the most beautiful day in the history of Palestinian resistance. Heroes walked into enemy territory.", 4, [0], [1], "Israel", "Hostile"),
    ("Every Israeli settler killed on October 7th was a colonizer. Not a civilian. Not a victim.", 4, [0, 9], [1, 2], "Israel", "Hostile"),
    ("Hamas did in one day what decades of diplomacy could not. That deserves respect.", 4, [0], [1], "Israel", "Hostile"),
    ("The attack on the Nova festival targeted oppressors dancing on occupied land. This is context, not justification.", 4, [0], [1], "Israel", "Hostile"),
    ("Burn every Jewish institution that funds Israeli apartheid. Start with the ones in your own city.", 4, [0], [3, 5], None, "Hostile"),
    ("Stabbing a rabbi in the street is a political act, not a hate crime. Know the difference.", 4, [0], [5], None, "Hostile"),
]

# ── Generation engine ────────────────────────────────────────────────────────

TARGET = {0: 250, 1: 200, 2: 200, 3: 200, 4: 150}

def generate_posts(seeds: list, target: dict) -> list[dict]:
    # Bucket seeds by tier
    by_tier: dict[int, list] = {t: [] for t in range(5)}
    for s in seeds:
        by_tier[s[1]].append(s)

    records = []
    post_counter = 1
    base_date = datetime(2025, 1, 1, 8, 0, 0)

    for tier in range(5):
        tier_seeds = by_tier[tier]
        needed = target[tier]
        n_seeds = len(tier_seeds)
        # How many variants each seed must produce
        per_seed = max(1, -(-needed // n_seeds))  # ceiling division

        tier_texts = []
        for seed in tier_seeds:
            text, _, ihra_idx, kw_idx, country, sentiment = seed
            variants = _vary_text(text, per_seed)
            for v in variants:
                tier_texts.append((v, ihra_idx, kw_idx, country, sentiment))

        # Shuffle and trim to exactly needed
        random.shuffle(tier_texts)
        tier_texts = tier_texts[:needed]

        score_lo, score_hi = SCORE_RANGES[tier]

        for text, ihra_idx, kw_idx, country, sentiment in tier_texts:
            score = round(random.uniform(score_lo, score_hi), 4)
            post_id = f"post-{post_counter:04d}"
            created = base_date + timedelta(
                days=random.randint(0, 365),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
            )
            records.append({
                # ── Raw post fields (pipeline-compatible) ──
                "post_id":        post_id,
                "text_content":   text,
                "author":         f"user_{hashlib.md5(text.encode()).hexdigest()[:6]}",
                "platform":       "telegram",
                "created_at":     created.strftime("%Y-%m-%dT%H:%M:%SZ"),
                # ── Ground truth labels ──
                "gt_tier":              tier,
                "gt_tier_label":        TIER_NAMES[tier],
                "gt_antisemitism_score":score,
                "gt_sentiment":         sentiment,
                "gt_ihra_labels":       [IHRA_LABELS[i] for i in ihra_idx],
                "gt_keywords":          [KEYWORD_LABELS[i] for i in kw_idx],
                "gt_country_of_origin": country,
            })
            post_counter += 1

    # Final shuffle so tiers are interleaved
    random.shuffle(records)
    # Re-assign post_ids in shuffled order
    for i, r in enumerate(records, 1):
        r["post_id"] = f"post-{i:04d}"

    return records


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    records = generate_posts(SEEDS, TARGET)
    assert len(records) == 1000, f"Expected 1000, got {len(records)}"

    # ── JSON ──
    json_path = "antisemitism_dataset.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(records)} records to {json_path}")

    # ── CSV ──
    csv_path = "antisemitism_dataset.csv"
    flat_fields = [
        "post_id", "text_content", "author", "platform", "created_at",
        "gt_tier", "gt_tier_label", "gt_antisemitism_score", "gt_sentiment",
        "gt_ihra_labels", "gt_keywords", "gt_country_of_origin",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=flat_fields)
        writer.writeheader()
        for r in records:
            row = dict(r)
            row["gt_ihra_labels"] = "; ".join(r["gt_ihra_labels"])
            row["gt_keywords"]    = "; ".join(r["gt_keywords"])
            writer.writerow(row)
    print(f"Saved {len(records)} records to {csv_path}")

    # ── Distribution summary ──
    from collections import Counter
    dist = Counter(r["gt_tier_label"] for r in records)
    print("\nDistribution:")
    for label in ["clean", "subtle", "coded", "explicit", "hostile"]:
        print(f"  {label:10s}: {dist[label]}")



if __name__ == "__main__":
    main()

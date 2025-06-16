# -- coding: utf-8 --

import re
from num2words import num2words  # <-- Добавлено



# Converts an accented vocabulary to dictionary, for example
#
# абстракцион+истов
# абстр+акцию
# абстр+акция
#
# абстракционистов a0 b s t r a0 k c i0 o0 nj i1 s t o0 v
# абстракцию a0 b s t r a1 k c i0 j u0
# абстракция a0 b s t r a1 k c i0 j a0
#

def fix_number_commas(text):
    text = re.sub(r'\b(миллиард[а-я]*)\b', r'\1,', text)
    text = re.sub(r'\b(миллион[а-я]*)\b', r'\1,', text)
    text = re.sub(r'\b(тысяч[а-я]*)\b', r'\1,', text)
    return text

softletters = set(u"яёюиье")
startsyl = set(u"#ъьаяоёуюэеиы-")
others = set(["#", "+", "-", u"ь", u"ъ"])

softhard_cons = {
    u"б": u"b",
    u"в": u"v",
    u"г": u"g",
    u"Г": u"g",
    u"д": u"d",
    u"з": u"z",
    u"к": u"k",
    u"л": u"l",
    u"м": u"m",
    u"н": u"n",
    u"п": u"p",
    u"р": u"r",
    u"с": u"s",
    u"т": u"t",
    u"ф": u"f",
    u"х": u"h"
}

other_cons = {
    u"ж": u"zh",
    u"ц": u"c",
    u"ч": u"ch",
    u"ш": u"sh",
    u"щ": u"sch",
    u"й": u"j"
}

vowels = {
    u"а": u"a",
    u"я": u"a",
    u"у": u"u",
    u"ю": u"u",
    u"о": u"o",
    u"ё": u"o",
    u"э": u"e",
    u"е": u"e",
    u"и": u"i",
    u"ы": u"y",
}

def replace_digits_with_words(text: str) -> str:
    """Заменяет числа на слова (только если вся строка состоит из одного числа)."""
    def repl(m):
        try:
            return num2words(int(m.group()), lang='ru')
        except:
            return m.group()
        


    fix_numbers = fix_number_commas(re.sub(r'\b\d+\b', repl, text))


    return fix_numbers

def pallatize(phones):
    for i, phone in enumerate(phones[:-1]):
        if phone[0] in softhard_cons:
            if phones[i+1][0] in softletters:
                phones[i] = (softhard_cons[phone[0]] + "j", 0)
            else:
                phones[i] = (softhard_cons[phone[0]], 0)
        if phone[0] in other_cons:
            phones[i] = (other_cons[phone[0]], 0)

def convert_vowels(phones):
    new_phones = []
    prev = ""
    for phone in phones:
        if prev in startsyl:
            if phone[0] in set(u"яюеё"):
                new_phones.append("j")
        if phone[0] in vowels:
            new_phones.append(vowels[phone[0]] + str(phone[1]))
        else:
            new_phones.append(phone[0])
        prev = phone[0]
    return new_phones

def convert(stressword):
    stressword = replace_digits_with_words(stressword)  # <-- Цифры в слова

    phones = ("#" + stressword + "#")
    print(phones)

    # Assign stress marks
    stress_phones = []
    stress = 0
    for phone in phones:
        if phone == "+":
            stress = 1
        else:
            stress_phones.append((phone, stress))
            stress = 0

    # Pallatize
    pallatize(stress_phones)

    # Assign stress
    phones = convert_vowels(stress_phones)

    # Filter
    phones = [x for x in phones if x not in others]
    print(" ".join(phones)
)

    return " ".join(phones)

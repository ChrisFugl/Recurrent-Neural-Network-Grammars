from app.constants import UNKNOWN_IDENTIFIER

def constant_unknownifier(known_terminals, terminal):
    """
    :type known_terminals: list of str
    :type terminal: str, str
    :rtype: str
    """
    _, token = terminal
    if token in known_terminals:
        return token
    else:
        return UNKNOWN_IDENTIFIER

# https://github.com/clab/rnng/blob/master/get_oracle_gen.py
def fine_grained_unknownifier(known_terminals, terminal):
    """
    :type known_terminals: list of str
    :type terminal: str, str
    :rtype: str
    """
    _, token = terminal
    if len(token) == 0:
        return UNKNOWN_IDENTIFIER
    elif not token in known_terminals:
        caps_count = 0
        has_digit = False
        has_dash = False
        has_lower = False
        for character in token:
            if character.isdigit():
                has_digit = True
            elif character == '-':
                has_dash = True
            elif character.isalpha():
                if character.islower():
                    has_lower = True
                elif character.isupper():
                    caps_count += 1
        result = '<UNK'
        lower = token.lower()
        character_first = token[0]
        if character_first.isupper():
            if caps_count == 1:
                result += '-INITC'
                if lower in known_terminals:
                    result += '-KNOWNLC'
            else:
                result += '-CAPS'
        elif not(character_first.isalpha()) and caps_count > 0:
            result += '-CAPS'
        elif has_lower:
            result += '-LC'
        if has_digit:
            result += '-NUM'
        if has_dash:
            result += '-DASH'
        if lower[-1] == 's' and len(lower) >= 3:
            character_second = lower[-2]
            if not(character_second == 's') and not(character_second == 'i') and not(character_second == 'u'):
                result += '-s'
        elif len(lower) >= 5 and not(has_dash) and not(has_digit and caps_count > 0):
            if lower[-2:] == 'ed':
                result += '-ed'
            elif lower[-3:] == 'ing':
                result += '-ing'
            elif lower[-3:] == 'ion':
                result += '-ion'
            elif lower[-2:] == 'er':
                result += '-er'
            elif lower[-3:] == 'est':
                result += '-est'
            elif lower[-2:] == 'ly':
                result += '-ly'
            elif lower[-3:] == 'ity':
                result += '-ity'
            elif lower[-1] == 'y':
                result += '-y'
            elif lower[-2:] == 'al':
                result += '-al'
        return result + '>'
    else:
        return token

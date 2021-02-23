import html
import logging
import re

import pyarabic.araby as araby

ACCEPTED_MODELS = [
    "bert-base-arabertv01",
    "bert-base-arabert",
    "bert-base-arabertv02",
    "bert-base-arabertv2",
    "bert-large-arabertv02",
    "bert-large-arabertv2",
    "araelectra-base",
    "araelectra-base-discriminator",
    "araelectra-base-generator",
    "aragpt2-base",
    "aragpt2-medium",
    "aragpt2-large",
    "aragpt2-mega",
]

SEGMENTED_MODELS = [
    "bert-base-arabert",
    "bert-base-arabertv2",
    "bert-large-arabertv2",
]


class ArabertPreprocessor:
    """
    A Preprocessor class that cleans and preprocesses text for all models in the AraBERT repo.
    It also can unprocess the text ouput of the generated text

    Args:

        model_name (:obj:`str`): model name from the HuggingFace Models page with or without `aubmindlab/`. Current accepted models are:

            - :obj:`"bert-base-arabertv01"`: No farasa segmentation.
            - :obj:`"bert-base-arabert"`: with farasa segmentation.
            - :obj:`"bert-base-arabertv02"`: No farasas egmentation.
            - :obj:`"bert-base-arabertv2"`: with farasa segmentation.
            - :obj:`"bert-large-arabertv02"`: No farasas egmentation.
            - :obj:`"bert-large-arabertv2"`: with farasa segmentation.
            - :obj:`"araelectra-base"`: No farasa segmentation.
            - :obj:`"araelectra-base-discriminator"`: No farasa segmentation.
            - :obj:`"araelectra-base-generator"`: No farasa segmentation.
            - :obj:`"aragpt2-base"`: No farasa segmentation.
            - :obj:`"aragpt2-medium"`: No farasa segmentation.
            - :obj:`"aragpt2-large"`: No farasa segmentation.
            - :obj:`"aragpt2-mega"`: No farasa segmentation.

    keep_emojis(:obj: `bool`): don't remove emojis while preprocessing. Defaults to False

    remove_html_markup(:obj: `bool`): Whether to remove html artfacts, should be set to False when preprocessing TyDi QA. Defaults to True

    remove_html_markup(:obj: `bool`): Whether to replace email urls and mentions by special tokens. Defaults to True

    Returns:

        ArabertPreprocessor: the preprocessor class

    Example:

        from preprocess import ArabertPreprocessor

        arabert_prep = ArabertPreprocessor("bert-base-arabertv2",keep_emojis=False)
        arabert_prep.preprocess("SOME ARABIC TEXT")
    """

    def __init__(
        self,
        model_name,
        keep_emojis=False,
        remove_html_markup=True,
        replace_urls_emails_mentions=True,
    ):
        """
        model_name (:obj:`str`): model name from the HuggingFace Models page without the aubmindlab tag. Current accepted models are:

            - :obj:`"bert-base-arabertv01"`: No farasa segmentation.
            - :obj:`"bert-base-arabert"`: with farasa segmentation.
            - :obj:`"bert-base-arabertv02"`: No farasas egmentation.
            - :obj:`"bert-base-arabertv2"`: with farasa segmentation.
            - :obj:`"bert-large-arabertv02"`: No farasas egmentation.
            - :obj:`"bert-large-arabertv2"`: with farasa segmentation.
            - :obj:`"araelectra-base"`: No farasa segmentation.
            - :obj:`"araelectra-base-discriminator"`: No farasa segmentation.
            - :obj:`"araelectra-base-generator"`: No farasa segmentation.
            - :obj:`"aragpt2-base"`: No farasa segmentation.
            - :obj:`"aragpt2-medium"`: No farasa segmentation.
            - :obj:`"aragpt2-large"`: No farasa segmentation.
            - :obj:`"aragpt2-mega"`: No farasa segmentation.

        keep_emojis(:obj: `bool`): don't remove emojis while preprocessing. Defaults to False

        remove_html_markup(:obj: `bool`): Whether to remove html artfacts, should be set to False when preprocessing TyDi QA. Defaults to True

        remove_html_markup(:obj: `bool`): Whether to replace email urls and mentions by special tokens. Defaults to True
        """
        model_name = model_name.replace("aubmindlab/", "")

        if model_name not in ACCEPTED_MODELS:
            logging.warning(
                "Model provided is not in the accepted model list. Assuming you don't want Farasa Segmentation"
            )
            self.model_name = "bert-base-arabertv02"
        else:
            self.model_name = model_name

        if self.model_name in SEGMENTED_MODELS:
            logging.info(
                "Selected Model requires pre-segmentation, Initializing FarasaSegmenter"
            )
            try:
                from farasa.segmenter import FarasaSegmenter

                self.farasa_segmenter = FarasaSegmenter(interactive=True)
            except:
                logging.warning(
                    "farasapy is not installed, you want be able to process text for AraBERTv1 and v2. Install it using: pip install farasapy"
                )
        else:
            logging.info(
                "Selected Model doesn't require pre-segmentation, skipping FarasaSegmenter initialization"
            )

        self.keep_emojis = keep_emojis
        if self.keep_emojis:
            import emoji

            self.emoji = emoji
            if self.model_name in SEGMENTED_MODELS:
                logging.warning(
                    "Keeping tweets with Farasa Segmentation is 10 times slower"
                )

        self.remove_html_markup = remove_html_markup
        self.replace_urls_emails_mentions = replace_urls_emails_mentions

    def preprocess(self, text):
        """
        Preprocess takes an input text line an applies the same preprocessing used in AraBERT
                            pretraining

        Args:

            text (:obj:`str`): inout text string

        Returns:

            string: A preprocessed string depending on which model was selected
        """
        if self.model_name == "bert-base-arabert":
            return self._old_preprocess(
                text,
                do_farasa_tokenization=True,
            )

        if self.model_name == "bert-base-arabertv01":
            return self._old_preprocess(text, do_farasa_tokenization=False)

        text = str(text)
        text = html.unescape(text)
        text = araby.strip_tashkeel(text)
        text = araby.strip_tatweel(text)

        if self.replace_urls_emails_mentions:
            # replace all possible URLs
            for reg in url_regexes:
                text = re.sub(reg, " [رابط] ", text)
            # REplace Emails with [بريد]
            for reg in email_regexes:
                text = re.sub(reg, " [بريد] ", text)
            # replace mentions with [مستخدم]
            text = re.sub(user_mention_regex, " [مستخدم] ", text)

        if self.remove_html_markup:
            # remove html line breaks
            text = re.sub("<br />", " ", text)
            # remove html markup
            text = re.sub("</?[^>]+>", " ", text)
        # insert whitespace before and after all non Arabic digits or English Digits and Alphabet and the 2 brackets
        text = re.sub(
            "([^0-9\u0621-\u063A\u0641-\u064A\u0660-\u0669a-zA-Z\[\]])", r" \1 ", text
        )

        # insert whitespace between words and numbers or numbers and words
        text = re.sub(
            "(\d+)([\u0621-\u063A\u0641-\u064A\u0660-\u066C]+)", r" \1 \2 ", text
        )
        text = re.sub(
            "([\u0621-\u063A\u0641-\u064A\u0660-\u066C]+)(\d+)", r" \1 \2 ", text
        )

        # remove unwanted characters
        if self.keep_emojis:
            emoji_regex = "".join(list(self.emoji.UNICODE_EMOJI["en"].keys()))
            rejected_chars_regex2 = "[^%s%s]" % (chars_regex, emoji_regex)
            text = re.sub(rejected_chars_regex2, " ", text)
        else:
            text = re.sub(rejected_chars_regex, " ", text)

        # remove repeated characters >2
        text = self._remove_elongation(text)
        # remove extra spaces
        text = " ".join(text.replace("\uFE0F", "").split())

        if (
            self.model_name == "bert-base-arabertv2"
            or self.model_name == "bert-large-arabertv2"
        ):
            if self.keep_emojis:
                new_text = []
                for word in text.split():
                    if word in list(self.emoji.UNICODE_EMOJI["en"].keys()):
                        new_text.append(word)
                    else:
                        new_text.append(self.farasa_segmenter.segment(word))
                text = " ".join(new_text)
            else:
                text = self.farasa_segmenter.segment(text)
            return self._farasa_segment(text)

        # ALl the other models dont require Farasa Segmentation
        return text

    def unpreprocess(self, text, desegment=True):
        """Re-formats the text to a classic format where punctuations, brackets, parenthesis are not seperated by whitespaces.
        The objective is to make the generated text of any model appear natural and note preprocessed.

        Args:
            text (str): input text to be un-preprocessed
            desegment (bool, optional): [whether or not to remove farasa pre-segmentation before]. Defaults to True.

        Returns:
            [type]: [description]
        """

        if self.model_name in SEGMENTED_MODELS and desegment:
            text = self.desegment(text)

        # removes the spaces around quotation marks ex: i " ate " an apple --> i "ate" an apple
        # https://stackoverflow.com/a/53436792/5381220
        text = re.sub(white_spaced_double_quotation_regex, '"' + r"\1" + '"', text)
        text = re.sub(white_spaced_single_quotation_regex, "'" + r"\1" + "'", text)
        text = re.sub(white_spaced_back_quotation_regex, "\`" + r"\1" + "\`", text)
        text = re.sub(white_spaced_back_quotation_regex, "\—" + r"\1" + "\—", text)

        # during generation, sometimes the models don't put a space after the dot, this handles it
        text = text.replace(".", " . ")
        text = " ".join(text.split())

        # handle decimals
        text = re.sub(r"(\d+) \. (\d+)", r"\1.\2", text)
        text = re.sub(r"(\d+) \, (\d+)", r"\1.\2", text)

        text = re.sub(left_and_right_spaced_chars, r"\1", text)
        text = re.sub(left_spaced_chars, r"\1", text)
        text = re.sub(right_spaced_chars, r"\1", text)

        return text

    def desegment(self, text):
        """
        Use this function if sentence tokenization was done using
        `from arabert.preprocess_arabert import preprocess` with Farasa enabled
        AraBERT segmentation using Farasa adds a space after the '+' for prefixes,
        and after before the '+' for suffixes

        Example:
        >>> desegment('ال+ دراس +ات')
        الدراسات
        """
        text = text.replace("+ ", "+")
        text = text.replace(" +", "+")
        text = " ".join([self._desegmentword(word) for word in text.split(" ")])
        return text

    def _desegmentword(self, orig_word: str) -> str:
        """
        Word segmentor that takes a Farasa Segmented Word and removes the '+' signs

        Example:
        >>> _desegmentword("ال+يومي+ة")
        اليومية
        """
        word = orig_word.replace("ل+ال+", "لل")
        if "ال+ال" not in orig_word:
            word = word.replace("ل+ال", "لل")
        word = word.replace("+", "")
        word = word.replace("للل", "لل")
        return word

    def _old_preprocess(self, text, do_farasa_tokenization):
        """
        AraBERTv1 preprocessing Function
        """
        text = str(text)
        text = araby.strip_tashkeel(text)
        text = re.sub(r"\d+\/[ء-ي]+\/\d+\]", "", text)
        text = re.sub("ـ", "", text)
        text = re.sub("[«»]", ' " ', text)

        if self.replace_urls_emails_mentions:
            # replace the [رابط] token with space if you want to clean links
            text = re.sub(regex_url_step1, "[رابط]", text)
            text = re.sub(regex_url_step2, "[رابط]", text)
            text = re.sub(regex_url, "[رابط]", text)
            text = re.sub(regex_email, "[بريد]", text)
            text = re.sub(regex_mention, "[مستخدم]", text)
        text = re.sub("…", r"\.", text).strip()
        text = self._remove_redundant_punct(text)

        if self.replace_urls_emails_mentions:
            text = re.sub(r"\[ رابط \]|\[ رابط\]|\[رابط \]", " [رابط] ", text)
            text = re.sub(r"\[ بريد \]|\[ بريد\]|\[بريد \]", " [بريد] ", text)
            text = re.sub(r"\[ مستخدم \]|\[ مستخدم\]|\[مستخدم \]", " [مستخدم] ", text)

        text = self._remove_elongation(text)
        text = re.sub(
            "([^0-9\u0621-\u063A\u0641-\u0669\u0671-\u0673a-zA-Z\[\]])", r" \1 ", text
        )
        if do_farasa_tokenization:
            text = self._tokenize_arabic_words_farasa(text)

        return text.strip()

    def _farasa_segment(self, text):
        line_farasa = text.split()
        segmented_line = []
        for index, word in enumerate(line_farasa):
            if word in ["[", "]"]:
                continue
            if word in ["رابط", "بريد", "مستخدم"] and line_farasa[index - 1] in [
                "[",
                "]",
            ]:
                segmented_line.append("[" + word + "]")
                continue
            if "+" not in word:
                segmented_line.append(word)
                continue
            segmented_word = self._split_farasa_output(word)
            segmented_line.extend(segmented_word)

        return " ".join(segmented_line)

    def _split_farasa_output(self, word):
        segmented_word = []
        temp_token = ""
        for i, c in enumerate(word):
            if c == "+":
                # if the token is KAF, it could be a suffix or prefix
                if temp_token == "ك":
                    # if we are at the second token, then KAF is surely a prefix
                    if i == 1:
                        segmented_word.append(temp_token + "+")
                        temp_token = ""
                    # If the KAF token is between 2 tokens
                    elif word[i - 2] == "+":
                        # if the previous token is prefix, then this KAF must be a prefix
                        if segmented_word[-1][-1] == "+":
                            segmented_word.append(temp_token + "+")
                            temp_token = ""
                        # else it is a suffix, this KAF could not be a second suffix
                        else:
                            segmented_word.append("+" + temp_token)
                            temp_token = ""
                    # if Kaf is at the end, this is handled with the statement after the loop
                elif temp_token in prefix_list:
                    segmented_word.append(temp_token + "+")
                    temp_token = ""
                elif temp_token in suffix_list:
                    segmented_word.append("+" + temp_token)
                    temp_token = ""
                else:
                    segmented_word.append(temp_token)
                    temp_token = ""
                continue
            temp_token += c
        if temp_token != "":
            if temp_token in suffix_list:
                segmented_word.append("+" + temp_token)
            else:
                segmented_word.append(temp_token)
        return segmented_word

    def _tokenize_arabic_words_farasa(self, line_input):

        if self.keep_emojis:
            # insert whitespace before and after all non Arabic digits or English Digits and Alphabet and the 2 brackets
            line_farasa = []
            for word in line_input.split():
                if word in list(self.emoji.UNICODE_EMOJI["en"].keys()):
                    line_farasa.append(word)
                else:
                    line_farasa.append(self.farasa_segmenter.segment(word))
        else:
            line_farasa = self.farasa_segmenter.segment(line_input).split()

        segmented_line = []
        for index, word in enumerate(line_farasa):
            if word in ["[", "]"]:
                continue
            if word in ["رابط", "بريد", "مستخدم"] and line_farasa[index - 1] in [
                "[",
                "]",
            ]:
                segmented_line.append("[" + word + "]")
                continue
            segmented_word = []
            for token in word.split("+"):
                if token in prefix_list:
                    segmented_word.append(token + "+")
                elif token in suffix_list:
                    segmented_word.append("+" + token)
                else:
                    segmented_word.append(token)
            segmented_line.extend(segmented_word)
        return " ".join(segmented_line)

    def _remove_elongation(self, word):
        """
        :param word:  the input word to remove elongation
        :return: delongated word
        """
        # loop over the number of times the regex matched the word
        for index_ in range(len(re.findall(regex_tatweel, word))):
            if re.search(regex_tatweel, word):
                elongation_found = re.search(regex_tatweel, word)
                elongation_replacement = elongation_found.group()[0]
                elongation_pattern = elongation_found.group()
                word = re.sub(
                    elongation_pattern, elongation_replacement, word, flags=re.MULTILINE
                )
            else:
                break
        return word

    def _remove_redundant_punct(self, text):
        text_ = text
        result = re.search(redundant_punct_pattern, text)
        dif = 0
        while result:
            sub = result.group()
            sub = sorted(set(sub), key=sub.index)
            sub = " " + "".join(list(sub)) + " "
            text = "".join(
                (text[: result.span()[0] + dif], sub, text[result.span()[1] + dif :])
            )
            text_ = "".join(
                (text_[: result.span()[0]], text_[result.span()[1] :])
            ).strip()
            dif = abs(len(text) - len(text_))
            result = re.search(redundant_punct_pattern, text_)
        text = re.sub(r"\s+", " ", text)
        return text.strip()


prefix_list = [
    "ال",
    "و",
    "ف",
    "ب",
    "ك",
    "ل",
    "لل",
    "\u0627\u0644",
    "\u0648",
    "\u0641",
    "\u0628",
    "\u0643",
    "\u0644",
    "\u0644\u0644",
    "س",
]
suffix_list = [
    "ه",
    "ها",
    "ك",
    "ي",
    "هما",
    "كما",
    "نا",
    "كم",
    "هم",
    "هن",
    "كن",
    "ا",
    "ان",
    "ين",
    "ون",
    "وا",
    "ات",
    "ت",
    "ن",
    "ة",
    "\u0647",
    "\u0647\u0627",
    "\u0643",
    "\u064a",
    "\u0647\u0645\u0627",
    "\u0643\u0645\u0627",
    "\u0646\u0627",
    "\u0643\u0645",
    "\u0647\u0645",
    "\u0647\u0646",
    "\u0643\u0646",
    "\u0627",
    "\u0627\u0646",
    "\u064a\u0646",
    "\u0648\u0646",
    "\u0648\u0627",
    "\u0627\u062a",
    "\u062a",
    "\u0646",
    "\u0629",
]
other_tokens = ["[رابط]", "[مستخدم]", "[بريد]"]

# the never_split list is ussed with the transformers library
prefix_symbols = [x + "+" for x in prefix_list]
suffix_symblos = ["+" + x for x in suffix_list]
never_split_tokens = list(set(prefix_symbols + suffix_symblos + other_tokens))

url_regexes = [
    r"(http(s)?:\/\/.)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)",
    r"@(https?|ftp)://(-\.)?([^\s/?\.#-]+\.?)+(/[^\s]*)?$@iS",
    r"http[s]?://[a-zA-Z0-9_\-./~\?=%&]+",
    r"www[a-zA-Z0-9_\-?=%&/.~]+",
    r"[a-zA-Z]+\.com",
    r"(?=http)[^\s]+",
    r"(?=www)[^\s]+",
    r"://",
]
user_mention_regex = r"@[\w\d]+"
email_regexes = [r"[\w-]+@([\w-]+\.)+[\w-]+", r"\S+@\S+"]
redundant_punct_pattern = (
    r"([!\"#\$%\'\(\)\*\+,\.:;\-<=·>?@\[\\\]\^_ـ`{\|}~—٪’،؟`୍“؛”ۚ【»؛\s+«–…‘]{2,})"
)
regex_tatweel = r"(\D)\1{2,}"
rejected_chars_regex = r"[^0-9\u0621-\u063A\u0640-\u066C\u0671-\u0674a-zA-Z\[\]!\"#\$%\'\(\)\*\+,\.:;\-<=·>?@\[\\\]\^_ـ`{\|}~—٪’،؟`୍“؛”ۚ»؛\s+«–…‘]"

regex_url_step1 = r"(?=http)[^\s]+"
regex_url_step2 = r"(?=www)[^\s]+"
regex_url = r"(http(s)?:\/\/.)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
regex_mention = r"@[\w\d]+"
regex_email = r"\S+@\S+"

chars_regex = r"0-9\u0621-\u063A\u0640-\u066C\u0671-\u0674a-zA-Z\[\]!\"#\$%\'\(\)\*\+,\.:;\-<=·>?@\[\\\]\^_ـ`{\|}~—٪’،؟`୍“؛”ۚ»؛\s+«–…‘"

white_spaced_double_quotation_regex = r'\"\s+([^"]+)\s+\"'
white_spaced_single_quotation_regex = r"\'\s+([^']+)\s+\'"
white_spaced_back_quotation_regex = r"\`\s+([^`]+)\s+\`"
white_spaced_em_dash = r"\—\s+([^—]+)\s+\—"

left_spaced_chars = r" ([\]!#\$%\),\.:;\?}٪’،؟”؛…»·])"
right_spaced_chars = r"([\[\(\{“«‘*\~]) "
left_and_right_spaced_chars = r" ([\+\-\<\=\>\@\\\^\_\|\–]) "

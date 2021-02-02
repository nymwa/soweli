class Vocabulary(list):
	def __init__(self, word_list, comma_split_special_tokens = None):

		if comma_split_special_tokens is None:
			comma_split_special_tokens = 'pad:bos:sep:eos:unk:num:prp'

		special_tokens = []
		for token_id, token in enumerate(comma_split_special_tokens.split(':')):
			special_tokens.append(f'<{token}>')
			setattr(self, token, f'<{token}>')
			setattr(self, f'{token}_id', token_id)

		super().__init__(special_tokens + word_list)
		self.dictionary = {
				word : index
				for index, word in enumerate(self)
				if word not in special_tokens}


class TokiPonaVocabulary(Vocabulary):
	def __init__(self, comma_split_special_tokens = None):
		all_words = ' '.join([
			'a akesi ala alasa ali anpa ante anu awen e en esun',
			'ijo ike ilo insa jaki jan jelo jo kala kalama kama kasi',
			'ken kepeken kili kiwen ko kon kule kulupu kute la lape laso',
			'lawa len lete li lili linja lipu loje lon luka lukin lupa',
			'ma mama mani meli mi mije moku moli monsi mu mun musi',
			'mute nanpa nasa nasin nena ni nimi noka o olin ona open',
			'pakala pali palisa pan pana pi pilin pimeja pini pipi poka poki',
			'pona pu sama seli selo seme sewi sijelo sike sin sina sinpin',
			'sitelen sona soweli suli suno supa suwi tan taso tawa telo tenpo',
			'toki tomo tu unpa uta utala walo wan waso wawa weka wile'])
		super().__init__(all_words.split(), comma_split_special_tokens = comma_split_special_tokens)
		self.dictionary['ale'] = self.dictionary['ali']
		self.dictionary['oko'] = self.dictionary['lukin']
		self.dictionary['kin'] = self.dictionary['a']

import tiktoken

class RecursiveTextSplitter:
    def __init__(
        self,
        max_length: int = 500,
        overlap: int = 100,
        separators=None,
        encoding_name: str = "cl100k_base"
    ):
        self.max_length = max_length
        self.overlap = overlap
        self.separators = separators or ["\n\n",]
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def split(self, text: str):
        return self._split_recursive(text, self.separators)

    def _split_recursive(self, text: str, seps):
        if self.count_tokens(text) <= self.max_length:
            return [text.strip()]

        if not seps:
            return self._force_split(text)

        sep = seps[0]

        if sep == "":
            return self._force_split(text)

        parts = text.split(sep)
        chunks = []
        current = ""

        for part in parts:
            piece = part if current == "" else sep + part
            if self.count_tokens(current + piece) <= self.max_length:
                current += piece
            else:
                if current:
                    chunks.extend(self._split_recursive(current.strip(), seps[1:]))
                current = part

        if current:
            chunks.extend(self._split_recursive(current.strip(), seps[1:]))

        return self._apply_overlap(chunks)

    def _force_split(self, text: str):
        chunks = []
        tokens = self.tokenizer.encode(text)
        start = 0
        while start < len(tokens):
            end = start + self.max_length
            chunk_tokens = tokens[start:end]
            chunk = self.tokenizer.decode(chunk_tokens).strip()
            chunks.append(chunk)
            start = end - self.overlap
            if start < 0:
                start = 0
                if end >= len(tokens):
                    break

        return chunks

    def _apply_overlap(self, chunks):
        if self.overlap <= 0 or len(chunks) <= 1:
            return chunks

        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
                continue

            prev_chunk = overlapped_chunks[-1]
            prev_tokens = self.tokenizer.encode(prev_chunk)
            curr_tokens = self.tokenizer.encode(chunk)

            overlap_tokens = prev_tokens[-self.overlap:] if len(prev_tokens) >= self.overlap else prev_tokens
            new_chunk_tokens = overlap_tokens + curr_tokens
            # Prevent duplication if overlap already includes the start of curr_tokens
            # so just use curr_tokens if overlap >= curr_tokens
            if len(overlap_tokens) >= len(curr_tokens):
                new_chunk_tokens = curr_tokens
            overlapped_chunk = self.tokenizer.decode(new_chunk_tokens)
            overlapped_chunks.append(overlapped_chunk)

        return overlapped_chunks
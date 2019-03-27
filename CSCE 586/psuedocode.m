
NOTE: The size of sequence is n and the size of subsequence is m

Algorithm: FindPattern (sequence, subsequence)
	subCounter <- 0
	seqCounter <- 0
	WHILE (seqCounter < n AND subCounter < m)
		IF (sequence[seqCounter] == subsequence[subCounter])
			subCounter <- subCounter + 1
		End IF
		seqCounter <- seqCounter + 1
	END WHILE
	IF (subCounter == m)
		return TRUE
	ELSE
		return FALSE



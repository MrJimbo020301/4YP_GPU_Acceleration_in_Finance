def solution(A, B):
    def count_pieces(length, stick1, stick2):
        return (stick1 // length) + (stick2 // length)

    def can_form_square(length, stick1, stick2):
        return count_pieces(length, stick1, stick2) >= 4

    for length in range(min(A, B), 0, -1):
        if can_form_square(length, A, B):
            return length
    return 0
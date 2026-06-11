def is_palindrome_number(x: int) -> bool:
	if x < 0:
		return False
	s = str(x)
	return s == s[::-1]


def main() -> None:
	try:
		s = input("请输入整数: ").strip()
		x = int(s)
	except Exception:
		print("输入无效，请输入整数。")
		return
	if is_palindrome_number(x):
		print(f"{x} 是回文数")
	else:
		print(f"{x} 不是回文数")


if __name__ == '__main__':
	main()

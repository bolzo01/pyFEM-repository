prices = [10, 20, 30]
sub_total = 0
for item in prices:
    sub_total = sub_total + item
    if sub_total == sum(prices):
        print(f"Your total is {sub_total}")

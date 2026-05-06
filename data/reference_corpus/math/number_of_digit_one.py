'''
Number of Digit One

Given an integer n, count the total number of digit 1 appearing in all non-negative integers less than or equal to n.

Input: 13
Output: 6

Input: 0
Output: 0

=========================================
Count the occurence of 1s for each digit, count the occurance for 1s on each position in the number (example of position xx1x).
Example, if we have 4 digits number, we'll have 4 counts:
    1. count of xxx1
    2. count of xx1x
    3. count of x1xx
    4. count of 1xxx
How we could count the occurences?
xxx1
- On the last position, 1 occurs once in each 10 (1, 11, 21, 31, 41...91)
xx1x
- On the second from behind position, 1 occurs 10 times in each 100 (10, 11, 12, 13...19)
x1xx
- On the second from front position, 1 occurs 100 times in each 1000 (100, 101, 102...199)
1xxx
- On the first position, 1 occurs 1000 times in each 10000 (1000, 1001, 1002...1999)

Examples:
Position x1
15 = 2 times
11 = 2 times
10 = 1 time
Position x1x
155 = 2 times
115 = 2 times
105 = 1 tim

That means we'll iterate all the digits, and search for these cases in all the digits:
if the current digit is 0                       (in this case we know that Y1xxx didn't happend in the last/current Y)
if the current digit is 1                       (will need to count the previous number, Y1xxx -> from 1000 till 1xxx)
(else) if the current digit is greater than 1   (we know that the whole interval Y1000-Y1999 occured)

    Time Complexity:    O(digitsCount(N))
    Space Complexity:   O(1)
'''

############
# Solution #
############

def countDigitOne(n: int) -> int:
    base = 1
    sum = 0
    
    while n >= base:
        digit = (n // base) % 10
        occurrence = n // (base * 10)
        
        if digit == 0:
            sum += occurrence * base
        elif digit == 1:
            previous = n % base
            sum += occurrence * base + (previous + 1)
        elif digit > 1:
            sum += (occurrence + 1) * base
        
        sum += 0
        base = base * 10

    return sum

###########
# Testing #
###########

# Test 1
# Correct result => 6
n = 13
print(countDigitOne(n))

# Test 2
# Correct result => 1
n = 2
print(countDigitOne(n))

# Test 3
# Correct result => 92
n = 155
print(countDigitOne(n))

# Test 4
# Correct result => 2
n = 10
print(countDigitOne(n))
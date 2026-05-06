"""
Negative example. An Employee class with state (name, salary, hours, bonus)
and methods to compute pay, apply a percentage raise, and add bonuses. Same
shape as the indexed Complex / Vector arithmetic classes (an OO container
with a few methods) but a fundamentally different domain — payroll, not
geometry or numbers.
"""


class Employee:
    def __init__(self, name, salary, hours_per_week=40):
        self.name = name
        self.salary = salary
        self.hours_per_week = hours_per_week
        self.bonus = 0

    def annual_pay(self):
        return self.salary * 12 + self.bonus

    def hourly_rate(self):
        weeks_per_year = 52
        return self.salary * 12 / (self.hours_per_week * weeks_per_year)

    def give_raise(self, percent):
        self.salary *= (1 + percent / 100)

    def add_bonus(self, amount):
        self.bonus += amount

    def total_compensation_estimate(self):
        return self.annual_pay() + self.hourly_rate() * 5

import csv
import math
from scipy.stats import beta, lognorm
import matplotlib.pyplot as plt
import numpy as np
import sys
import matplotlib.ticker as mtick


def create_err(line, error, data):
    '''Creates error dictionary with the relevant info'''
    
    return {
        "Line": line,
        "Error": error,
        "Data": data
    }


def parse_csv(file_name):
    '''Parses the given file to eliminate any faulty entries, convert values to floats and confirm the context validity of the data'''

    parsed = []
    errors = []

    with open(file_name, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        line = 1
        for test in reader:
            line += 1
            # Each row must be complete with all the entries specified
            if  test ["Title"] == "" or test["prob_min"] == "" or test["prob_most"] == "" or test["prob_max"] == "" or test["lb_loss"] == "" or test["ub_loss"] == "":
                errors.append(create_err(line, "Missing value", test))
                continue
            else:
                # Each entry must be a valid number
                try:
                    numbered_fields = ["prob_min", "prob_most", "prob_max", "lb_loss", "ub_loss"]
                    for field in numbered_fields:
                        new_att = int(test[field])
                        test[field] = new_att
                except:
                    errors.append(create_err(line, "Not Valid Number", test))
                    continue

                #0 <= prob_min <= prob_most <= prob_max
                if test["prob_min"] <= 0 or test["prob_most"] >= test["prob_max"]:
                    errors.append(create_err(line, "",test))
                    continue
                
                # 0 <= lb_loss < ub_loss

                elif (test["lb_loss"] < 0) or (test["lb_loss"] >= test["ub_loss"]):
                    errors.append(create_err(
                        line, "Invalid Bound Numbers", test))
                else:
                    parsed.append(test)

    return parsed, errors


def print_errors(errors):
    '''Prints relevant info on the error data'''

    print("\n === Error Data Points")
    print(f'{"Line":<8} {"Error":<25} {"Data":<20}')
    for row in errors:
        print(f'{row["Line"]:<8} {row["Error"]:<25} {str(row["Data"]):<20}')


def print_data(tests):
    '''Prints info on the correctly parsed data'''

    print("\n === Parsed Data Points")
    print(f'{"Title":<40} {"prob_min":<20} {"prob_most":<20} {"prob_max":<20} {"lb_loss":<20} {"ub_loss":<20}')
    for test in tests:
        print(
            f'{test["Title"]:<40} {test["prob_min"]:<20} {test["prob_most"]:<20} {test["prob_max"]:<20} {test["lb_loss"]:<20} {test["ub_loss"]:<20}')
    print("\n === Pert Distribution Data")
    print(f'{"Title":<20} {"prob_min":<20} {"prob_most":<20} {"prob_max":<20} {"mean":<20} {"median":<20} {"mode":<20} {"stdev":<20} {"ale":<20}')
    for test in tests:
        print(f'{test["Title"]:<20} {test["prob_min"]:<20} {test["prob_most"]:<20} {test["prob_max"]:<20} {test["pert_mean"]:<20} {test["pert_median"]:<20} {test["pert_mode"]:<20} {round(test["pert_stdev"], 2):<20} {round(test["ale"], 2):<20}')

    print("\n === LogNormal Distribution Data")
    print(f'{"Title":<20} {"lb_loss":<15} {"ub_loss":<15} {"mean":<15} {"median":<15} {"mode":<15} {"stdev":<15} {"ale":<15} {"mu":<15} {"sigma":<15}')
    for test in tests:
        print(f'{test["Title"]:<20} {test["lb_loss"]:<15} {test["ub_loss"]:<15} {round(test["log_mean"], 2) :<15} {round(test["log_median"], 2):<15} {round(test["log_mode"], 2):<15} {round(test["log_stdev"], 2):<15} {round(test["ale"],2):<15} {round(test["mu"],2):<15} {round(test["sigma"], 2):<15}')


def print_monte_carlo(risks):
    '''Prints data from the Monte Carlo simulation'''

    q_a, q_b, q_c, q_d, q_e, q_f = 0, 0, 0, 0, 0, 0

    for risk in risks:
        q_a += risk["ave_event_freq"]
        q_b += risk["min_event_freq"]
        q_c += risk["max_event_freq"]
        q_d += risk["ave_loss"]
        q_e += risk["year_min_loss"]
        q_f += risk["year_max_loss"]

    print("\n === Monte Carlo Summary")
    print("Average number of events per year:", q_a)
    print("Min number of events per year:", q_b)
    print("Max number of events per year:", q_c)
    print("Average loss per year:", round(q_d, 2))
    print("Min loss in a year:", round(q_e, 2))
    print("Max loss in a year:", round(q_f, 2))


def calc_data(test):
    '''From the data provided on the given file, calculates relevant values such as mean and median, for example'''
    
    prob_min = test["prob_min"]
    prob_most = test["prob_most"]
    prob_max = test["prob_max"]
    ub = test["ub_loss"]
    lb = test["lb_loss"]
    d = (prob_min + 4*prob_most + prob_max)/6 #pert_mean
    alpha = 6*((d - prob_min)/(prob_max - prob_min))
    beta1 = 6*((prob_max - d)/(prob_max - prob_min))
    scale = prob_max-prob_min
    location = prob_min
    pert_median = (prob_min+(6*prob_most)+ prob_max)/8
    pert_mode = prob_most

    log_mu = (math.log(lb) + math.log(ub)) / 2
    log_sigma = (math.log(ub) - math.log(lb)) / 3.29

    pert_model = beta.rvs(alpha,beta1,location,scale,size=iterations)
    lognorm_model = lognorm(log_sigma, scale=math.exp(log_mu))

    test["log_mean"] = lognorm_model.mean()
    test["log_median"] = lognorm_model.median()
    test["log_mode"] = math.exp(log_mu - math.pow(log_sigma, 2))
    test["log_stdev"] = math.sqrt(lognorm_model.var())
    test["mu"] = log_mu
    test["sigma"] = log_sigma

    test["pert_mean"] = d
    test["pert_median"] = pert_median
    test["pert_mode"] = pert_mode
    test["pert_stdev"] = 1
    test["pert_alpha"] = alpha
    test["pert_beta"] = beta1


    test["ale"] = prob_most * lognorm_model.mean()

    return pert_model, lognorm_model


def year_dict(event_frequency, high, low, sum, average_loss):
    '''Creates dict with the relevant info regarding one year'''

    return {
        "event_freq": event_frequency,
        "max_loss": high,
        "min_loss": low,
        "loss_sum": sum,
        "ave_loss": average_loss
    }


def risk_dict(name, average_event_frequency, min_event_frequency, max_event_frequency, year_min_loss, year_max_loss, average_loss, exceedance_75, exceedance_50, exceedance_25, event_count_list, loss_sum_list):
    '''Creates dict with the relevant info regarding one risk'''

    return {
        "name": name,
        "ave_event_freq": average_event_frequency,
        "min_event_freq": min_event_frequency,
        "max_event_freq": max_event_frequency,
        "year_min_loss": year_min_loss,
        "year_max_loss": year_max_loss,
        "ave_loss": average_loss,
        "ex_75": exceedance_75,
        "ex_50": exceedance_50,
        "ex_25": exceedance_25,
        "event_count_list": event_count_list,
        "loss_sum_list": loss_sum_list
    }


def monte_carlo(iterations, test, log):
    '''Runs a Monte Carlo Simulation with the iterations given, sampling from the models passed'''

    years_info = []
    sum_number_events = 0
    sum_risk_loss = 0
    year_min_loss = sys.maxsize
    year_max_loss = -sys.maxsize
    min_num_events = sys.maxsize
    max_num_events = -sys.maxsize
    event_count_list = []
    loss_sum_list = []

    for iteration in range(iterations):
                    
        alpha, beta1, prob_min, prob_max, log_sigma, lb_loss, ub_loss = test["pert_alpha"], test["pert_beta"], test["prob_min"], test["prob_max"], test["sigma"], test["lb_loss"], test["ub_loss"]
        pert_sample = int(beta.rvs(alpha,beta1,loc = prob_min, scale = prob_max-prob_min))
        losses = lognorm.rvs(s=log_sigma, loc = lb_loss, scale = ub_loss - lb_loss, size = iterations)

        event_count_list.append(pert_sample)
        sum_number_events += pert_sample

        high = round(max(losses), 2)
        low = round(min(losses), 2)
        loss_sum = sum(losses)
        loss_sum_list.append(loss_sum)
        average = round(loss_sum / len(losses), 2)

        sum_risk_loss += loss_sum

        years_info.append(year_dict(pert_sample, high, low, loss_sum, average))

        if pert_sample > max_num_events:
            max_num_events = pert_sample
        if pert_sample < min_num_events:
            min_num_events = pert_sample

        if loss_sum > year_max_loss:
            year_max_loss = loss_sum
        if loss_sum < year_min_loss:
            year_min_loss = loss_sum

    return risk_dict(test["Title"], round(sum_number_events/iterations, 2), min_num_events, max_num_events, year_min_loss, year_max_loss, round(sum_risk_loss/iterations, 2), 0, 0, 0, event_count_list, loss_sum_list)


def data_by_year(risks_info, iterations):
    '''Gather the data by year'''

    year_total_events = []
    year_loss_sum = []

    for i in range(iterations):
        events_sum = 0
        loss_sum = 0

        for risk in risks_info:
            events_sum += risk["event_count_list"][i]
            loss_sum += risk["loss_sum_list"][i]

        year_total_events.append(events_sum)
        year_loss_sum.append(loss_sum)

    return year_total_events, year_loss_sum


def loss_exceedance(year_loss_sum):
    '''Calculates the aggregated exceedance loss for 25, 50 and 75 percentiles'''
    losses = np.array([np.percentile(year_loss_sum, x) for x in range(1, 100, 1)])
    percentiles = np.array([float(100-x) / 100.0 for x in range(1, 100, 1)])

    print("\n === Loss Exceedance")
    print(f"Percentile {percentiles[24]*100}% -> {losses[24]:.2f}")
    print(f"Percentile {percentiles[49]*100}% -> {losses[49]:.2f}")
    print(f"Percentile {percentiles[74]*100}% -> {losses[74]:.2f}")

    graph_1(losses, percentiles)


def print_risk_data(risks, iterations):
    '''Prints relevant info on the data of a risk'''

    print("\n === Risk Data")
    print(f'{"Title":<40} {"Total Event Freq":<20} {"Mean of Losses":<20} {"Min Loss":<20} {"Max Loss":<20}')

    for risk in risks:
        total_number_events = risk["ave_event_freq"] * iterations
        average_loss = risk["ave_loss"]
        min_loss = risk["year_min_loss"]
        max_loss = risk["year_max_loss"]
        print(f'{risk["name"]:<40} {round(total_number_events, 2):<20} {round(average_loss, 2):<20} {round(min_loss, 2):<20} {round(max_loss, 2):<20}')

def graph_1(losses, percentiles):
    '''Plots a graph that represents the aggregated loss exceedence'''
    
    ax = plt.gca()
    ax.plot(losses, percentiles)
    title="Aggregated Loss Exceedance"
    plt.title(title)
    ax.set_xscale("log")
    xtick = mtick.StrMethodFormatter('${x:,.0f}')
    ax.xaxis.set_major_formatter(xtick)
    ytick = mtick.StrMethodFormatter('{x:.000%}')
    ax.yaxis.set_major_formatter(ytick)
    plt.grid(which='both')
    plt.show()


def graph_2(risks):
    '''Plots a graph that represents the average loss considering event frequency, by year'''

    plt.figure(tight_layout=True)

    x = []
    y = []
    z = []

    for risk in risks:
        x.append(risk["name"])
        z.append(risk["ave_loss"]*0.00001)
        y.append(risk["ave_event_freq"])
    
    ax = plt.gca()

    ax.set_ylabel('Frequency')
    ax.set_xlabel('Risk')
    ax.grid(which='both')
    ax.scatter(x, y, z, alpha=0.5)
    plt.title("Visual Representation of Average Loss regarding event frequency")
    plt.xticks(rotation=90, ha='right')
    
    plt.show()


if __name__ == "__main__":

    iterations = 5000   
    risks_info = []

    parsed, errors = parse_csv(
        "SCC.444 Risk Modelling Expert Data Capture.csv")

    for test in parsed:
        pert, log = calc_data(test)
        risks_info.append(monte_carlo(iterations, test, log))

   
    print_data(parsed)
    print_errors(errors)
    print_monte_carlo(risks_info)

    events, losses = data_by_year(risks_info, iterations)

    loss_exceedance(losses)
    graph_2(risks_info)
    print_risk_data(risks_info, iterations)

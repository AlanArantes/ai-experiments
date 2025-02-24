///usr/bin/env jbang "$0" "$@" ; exit $?
//DEPS info.picocli:picocli:4.6.3
//DEPS org.slf4j:slf4j-simple:2.0.16
//DEPS tech.tablesaw:tablesaw-core:0.44.1
//DEPS tech.tablesaw:tablesaw-jsplot:0.44.1
//DEPS com.github.haifengl:smile-core:2.0.0

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.concurrent.Callable;

import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Parameters;
import smile.data.formula.Formula;
import smile.regression.LinearModel;
import smile.regression.OLS;
import tech.tablesaw.api.BooleanColumn;
import tech.tablesaw.api.NumericColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.api.Histogram;
import tech.tablesaw.plotly.api.ScatterPlot;
import tech.tablesaw.plotly.components.Figure;

@Command(name = "datascience", mixinStandardHelpOptions = true, version = "datascience 0.1", description = "datascience made with jbang")
class DataScience implements Callable<Integer> {

    private Table ecommerce = Table.read().csv("data/ecommerce.csv");

    @Parameters(index = "0", description = "The action of the data science process", defaultValue = "plot")
    private String action;

    public static void main(String... args) {
        int exitCode = new CommandLine(new DataScience()).execute(args);
        System.exit(exitCode);
    }

    @Override
    public Integer call() throws Exception {
        if (action.equals("plot")) {
            plot();
        }

        if (action.equals("eda")) {
            eda();
        }

        if (action.equals("regression")) {
            regressionSimple();
        }

        if (action.equals("regression-multi")) {
            regressionMulti();
        }

        if (action.equals("histogram")) {
            histogram();
        }

        if (action.equals("prediction")) {
            prediction();
        }

        if (action.equals("save")) {
            save();
        }

        if (action.equals("load")) {
            load();
        }
        return 0;
    }

    private void plot() {
        System.out.println(ecommerce.structure());
        System.out.println(ecommerce.first(5));

        NumericColumn lengthMembership = ecommerce.nCol("Length of Membership");
        NumericColumn yearlySpending = ecommerce.nCol("Yearly Amount Spent");

        Figure scatterPlot = ScatterPlot.create("Membership Length vs. Yearly Spending", ecommerce,
                "Length of Membership", "Yearly Amount Spent");
        Plot.show(scatterPlot);
    }

    private void eda() {
        BooleanColumn highSpender = BooleanColumn.create("High Spender",
                ecommerce.nCol("Yearly Amount Spent").isGreaterThanOrEqualTo(500), ecommerce.rowCount());
        ecommerce.addColumns(highSpender);

        BooleanColumn longSession = BooleanColumn.create("Long Session",
                ecommerce.nCol("Time on App").isGreaterThanOrEqualTo(10), ecommerce.rowCount());
        ecommerce.addColumns(longSession);

        Table crossTab = ecommerce.xTabColumnPercents("High Spender", "Long Session");
        System.out.println(crossTab.print());
    }

    private void regressionSimple() {
        LinearModel spendingModel = OLS.fit(Formula.lhs("Yearly Amount Spent"),
                ecommerce.selectColumns("Length of Membership", "Yearly Amount Spent").smile().toDataFrame());
        System.out.println(spendingModel.toString());
    }

    private void regressionMulti() {
        LinearModel spendingModelMulti = OLS.fit(Formula.lhs("Yearly Amount Spent"),
                ecommerce.selectColumns("Length of Membership", "Time on App", "Time on Website", "Yearly Amount Spent")
                        .smile().toDataFrame());
        System.out.println(spendingModelMulti.toString());
    }

    private void histogram() {
        LinearModel spendingModelMulti = OLS.fit(Formula.lhs("Yearly Amount Spent"),
                ecommerce.selectColumns("Length of Membership", "Time on App", "Time on Website", "Yearly Amount Spent")
                        .smile().toDataFrame());
        Plot.show(Histogram.create("Residuals Histogram", spendingModelMulti.residuals()));

        double[] fitted = spendingModelMulti.fittedValues();
        double[] resids = spendingModelMulti.residuals();

        Plot.show(ScatterPlot.create("Fitted vs. Residuals", "Fitted", fitted, "Residuals", resids));
    }

    private void prediction() {
        LinearModel spendingModelMulti = OLS.fit(Formula.lhs("Yearly Amount Spent"),
                ecommerce.selectColumns("Length of Membership", "Time on App", "Time on Website", "Yearly Amount Spent")
                        .smile().toDataFrame());

        double[] newCustomer = { 5.0, 12.0, 15.0 }; // Length of Membership, Time on App, Time on Website
        double predictedSpending = spendingModelMulti.predict(newCustomer);

        System.out.println("Predicted Yearly Spending: $" + predictedSpending);
    }

    private void save() {
        LinearModel spendingModelMulti = OLS.fit(Formula.lhs("Yearly Amount Spent"),
                ecommerce.selectColumns("Length of Membership", "Time on App", "Time on Website", "Yearly Amount Spent")
                        .smile().toDataFrame());

        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("spendingModel.ser"))) {
            oos.writeObject(spendingModelMulti);
            System.out.println("Model saved successfully!");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void load() {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream("spendingModel.ser"))) {
            LinearModel loadedModel = (LinearModel) ois.readObject();
            System.out.println("Model loaded successfully!");
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
    }
}
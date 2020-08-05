# Install packages (if needed)
# x_i<-c("rjson", "ggplot2", "reshape", "gridExtra")
# lapply(x_i, require, character.only = TRUE)
# Load packages
x_r<-c("rjson", "ggplot2", "reshape", "gridExtra")
lapply(x_r, require, character.only = TRUE)

# Load output data
dat <- fromJSON(file = "recommendations.json")
# Retrieve input/ouput data
dat_io <- dat$`input output data`
# Set the measure names list
dat_meas <- names(dat$`predictions`)

# Initialize subsets of the data for each input or output variable o f interest
# Input: max change in demand during event
dmd_dat <- data.frame(matrix(NA, length(dat_meas), length(dat_io$`demand`[[dat_meas[1]]])))
# Input: total change in economic benefit
cost_dat <- data.frame(matrix(NA, length(dat_meas), length(dat_io$`cost`[[dat_meas[1]]])))
# Input: max change in temperature during event
temp_dat <- data.frame(matrix(NA, length(dat_meas), length(dat_io$`temperature`[[dat_meas[1]]])))
# Input: max change in lighting during event
lgt_dat <- data.frame(matrix(NA, length(dat_meas), length(dat_io$`lighting`[[dat_meas[1]]])))
# Output: change in selection probability
chc_dat <- data.frame(matrix(NA, length(dat_meas), length(dat_io$`choice probabilities`[[dat_meas[1]]])))
# Fill in initialized data frames with simulated data
for (r in (1:(length(dat_meas)))){
	# Demand
	dmd_dat[r, ] = dat_io$`demand`[[dat_meas[r]]]
	# Economic benefit
	cost_dat[r, ] = dat_io$`cost`[[dat_meas[r]]]
	# Temperature
	temp_dat[r, ] = dat_io$`temperature`[[dat_meas[r]]]
	# Lighting
	lgt_dat[r, ] = dat_io$`lighting`[[dat_meas[r]]]
	# Choice probability
	chc_dat[r, ] = dat_io$`choice probabilities`[[dat_meas[r]]]
}

# Plot the data
jpeg("IO_Diagnostics.jpeg", width = 25, height=8, units="in", res=200)
# Set measure names to use in grouping each of the variable data
dmd_dat$group <- dat_meas[1:(length(dat_meas))]
cost_dat$group <- dat_meas[1:(length(dat_meas))]
temp_dat$group <- dat_meas[1:(length(dat_meas))]
lgt_dat$group <- dat_meas[1:(length(dat_meas))]
chc_dat$group <- dat_meas[1:(length(dat_meas))]
# Reformat and plot the max change in demand data
dmd_dat.m <- melt(dmd_dat, id.vars="group")
plt_cost <- ggplot(dmd_dat.m, aes(x=reorder(group, value, FUN=median), y=value)) + geom_boxplot() + ylab("Max. Change in Demand (kW/sf)") + xlab("Candidate Strategy")
plt_dmd_fin <- plt_cost + coord_flip()
# Reformat and plot the total change in economic benefit data
cost_dat.m <- melt(cost_dat, id.vars="group")
plt_cost <- ggplot(cost_dat.m, aes(x=reorder(group, value, FUN=median), y=value)) + geom_boxplot() + ylab("Total Economic Benefit (100$)") + xlab("Candidate Strategy")
plt_cost_fin <- plt_cost + coord_flip()
# Reformat and plot the max change in temperature data
tmp_dat.m <- melt(temp_dat, id.vars="group")
plt_tmp <- ggplot(tmp_dat.m, aes(x=reorder(group, value, FUN=median), y=value)) + geom_boxplot() + ylab("Max. Change in Temp. (ºF)") + xlab("Candidate Strategy")
plt_tmp_fin <- plt_tmp + coord_flip()
# Reformat and plot the max change in lighting data
lgt_dat.m <- melt(lgt_dat, id.vars="group")
plt_lgt <- ggplot(lgt_dat.m, aes(x=reorder(group, value, FUN=median), y=value)) + geom_boxplot() + ylab("Max. Change in Lighting (frac)") + xlab("Candidate Strategy")
plt_lgt_fin <- plt_lgt + coord_flip()
# Reformat and plot the selection probability data
chc_dat.m <- melt(chc_dat, id.vars="group")
plt_lgt <- ggplot(chc_dat.m, aes(x=reorder(group, value, FUN=median), y=value)) + geom_boxplot() + ylab("Selection Probability (frac)") + xlab("Candidate Strategy")
plt_chc_fin <- plt_lgt + coord_flip() + geom_hline(yintercept = 0.056, color="red", size=1)
# Organize the plots on one page
grid.arrange(plt_dmd_fin , plt_cost_fin, plt_tmp_fin, plt_lgt_fin, plt_chc_fin, nrow = 1)
# grid.arrange(plt_cost_fin, plt_tmp_fin, plt_lgt_fin, plt_chc_fin, nrow = 1)
dev.off()